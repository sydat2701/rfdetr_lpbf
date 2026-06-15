#!/usr/bin/env python3
"""
RF-DETR Training Script for L-PBF Anomaly Detection.

Splits data by machine ID (--val-object, --test-object) to ensure
generalization across different print beds / camera setups.
Supports RFDETR2XLarge via the rfdetr_plus extension.
"""

import sys
import os

# Use local rfdetr/rfdetr_plus copies instead of site-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.datasets.coco import CocoDetection, build_roboflow_from_coco, make_coco_transforms
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize


VARIANT_STRIDE = {
    "nano": 32,     # patch=16, windows=2
    "small": 32,
    "medium": 32,
    "large": 32,
    "xlarge": 20,   # patch=20, windows=1
    "2xlarge": 40,  # patch=20, windows=2
    "base": 56,     # patch=14, windows=4
}

VARIANT_DEFAULT_RES = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "large": 704,
    "xlarge": 700,
    "2xlarge": 880,
    "base": 560,
}


CLASS_NAMES = [
    "shortage",
    "peel off",
    "streaking",
    "hopping",
    "hole",
    "powder residue",
    "spatter",
]

# Order must match CLASS_NAMES
CAT_NAME_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def compute_class_alpha(annotation_path: str, val_objects: set, test_objects: set, exclude_objects: set,
                        alpha_base: float = 0.25, power: float = 0.5) -> list[float]:
    """Compute per-class focal loss alpha weights inversely proportional to class frequency.

    Rare classes get higher alpha (more weight on positive examples), common classes
    get lower alpha. The average alpha across classes is approximately alpha_base.

    Returns:
        List of alpha values per class in CLASS_NAMES order.
    """
    with open(annotation_path) as f:
        coco = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img_machine = {}
    for img in coco["images"]:
        m = re.search(r"(SI\d{4})", img["file_name"].upper())
        if m:
            img_machine[img["id"]] = m.group(1)

    # Count annotations per class, only for training machines
    train_machines = set(img_machine.values()) - val_objects - test_objects - exclude_objects
    class_counts = {name: 0 for name in CLASS_NAMES}
    for ann in coco["annotations"]:
        mach = img_machine.get(ann["image_id"])
        if mach in train_machines:
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name in class_counts:
                class_counts[cat_name] += 1

    total = sum(class_counts.values())
    n_classes = len(CLASS_NAMES)

    alphas = []
    for name in CLASS_NAMES:
        count = class_counts[name]
        if count == 0:
            alphas.append(0.5)
        else:
            weight = (total / (n_classes * count)) ** power
            alpha = alpha_base * weight
            alphas.append(max(0.01, min(0.99, alpha)))

    return alphas

# Machine-specific ROI points for perspective crop (4 corners of print bed).
# Each entry: [top-left, top-right, bottom-right, bottom-left] in (x, y).
MACHINE_ROIS = {
    "SI2028": [[250, 96], [1272, 116], [1272, 908], [182, 902]],
    "SI2674": [[190, 114], [1276, 122], [1272, 922], [132, 926]],
    "SI3073": [[176, 114], [1276, 132], [1276, 926], [96, 930]],
    "SI3074": [[200, 132], [1276, 136], [1272, 932], [136, 940]],
    "SI3186": [[245, 87], [1271, 107], [1271, 928], [150, 913]],
    "SI3397": [[114, 276], [903, 264], [1032, 1071], [6, 1100]],
    "SI3588": [[160, 306], [934, 324], [1046, 1206], [4, 1206]],
    "SI3781": [[203, 109], [1280, 134], [1280, 943], [123, 941]],
    "SI3783": [[240, 105], [1280, 122], [1280, 936], [154, 921]],
    "SI3785": [[233, 122], [1280, 114], [1280, 928], [166, 942]],
    "SI3794": [[241, 121], [1280, 137], [1280, 936], [142, 936]],
    "SI3803": [[221, 101], [1280, 106], [1280, 936], [142, 936]],
    "SI4222": [[250, 108], [1280, 122], [1280, 940], [162, 933]],
}


def get_machine_roi(file_path):
    """Look up ROI points for a machine ID in the given path. Returns None if not found."""
    for mid in MACHINE_ROIS:
        if mid in file_path:
            return np.array(MACHINE_ROIS[mid], dtype=np.float32)
    return None


def machine_based_crop(img_np, file_path):
    """Perspective-crop image to the machine's print bed region.

    Args:
        img_np: H×W×3 uint8 or float32 image array.
        file_path: Path containing machine ID (e.g. 'SI3073...').

    Returns:
        (cropped_img, transform_matrix) where cropped_img is the warped output
        and M is the 3×3 perspective transform. If no ROI is found, returns
        (img_np, identity_3x3).
    """
    pts = get_machine_roi(file_path)
    if pts is None:
        return img_np, np.eye(3, dtype=np.float32)

    width_a = np.linalg.norm(pts[2] - pts[3])
    width_b = np.linalg.norm(pts[1] - pts[0])
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(pts[1] - pts[2])
    height_b = np.linalg.norm(pts[0] - pts[3])
    max_h = int(max(height_a, height_b))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_np, M, (max_w, max_h))
    return warped, M


def warp_coco_bbox(bbox, M, orig_w, orig_h, crop_w, crop_h):
    """Warp a single COCO bbox [x, y, w, h] using perspective matrix M.

    Returns the new axis-aligned bbox [x', y', w', h'] clipped to crop bounds.
    """
    x, y, w, h = bbox
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1 = max(0, xs.min())
    y1 = max(0, ys.min())
    x2 = min(crop_w, xs.max())
    y2 = min(crop_h, ys.max())
    return [x1, y1, x2 - x1, y2 - y1]



def extract_object_id(file_name):
    """Extract 6-char machine ID (e.g. 'SI3074') from image filename."""
    match = re.search(r"(SI\d{4})", file_name.upper())
    return match.group(1) if match else None


def resolve_image_path(file_name, image_dir):
    """Find the actual image file on disk by trying multiple strategies."""
    basename = os.path.basename(file_name)
    candidate = os.path.join(image_dir, basename)
    if os.path.isfile(candidate):
        return candidate
    candidate2 = os.path.join("/mnt/1tb/XAM/", file_name)
    if os.path.isfile(candidate2):
        return candidate2
    if os.path.isfile(file_name):
        return file_name
    if os.path.isfile(basename):
        return basename
    return None


def find_exposure_pair(recoat_path):
    """Find the exposure image paired with a recoating image path."""
    dirname = os.path.dirname(recoat_path)
    basename = os.path.basename(recoat_path)
    expo_basename = basename.replace("recoating", "exposure")
    expo_path = os.path.join(dirname, expo_basename)
    if os.path.isfile(expo_path):
        return expo_path
    alt = os.path.join(dirname, "exposure_" + basename.replace("recoating_", ""))
    if os.path.isfile(alt):
        return alt
    return None


def prepare_dataset(args):
    """Load COCO JSON, split images by machine ID, copy to output structure."""
    output_data_dir = os.path.join(args.output, "data")

    skip_file = os.path.join(output_data_dir, ".prepared")
    if os.path.exists(skip_file) and not args.force:
        print(f"  Dataset already prepared at {output_data_dir}. Use --force to re-prepare.")
        return output_data_dir

    print("  Loading COCO annotations...")
    with open(args.annotation, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Categorize image IDs by machine ID
    machine_to_image_ids = defaultdict(list)
    image_id_to_info = {img["id"]: img for img in coco["images"]}
    image_path_cache = {}

    for img in coco["images"]:
        obj_id = extract_object_id(img["file_name"])
        if obj_id is None:
            print(f"  Warning: Could not extract machine ID from '{img['file_name']}', skipping")
            continue
        machine_to_image_ids[obj_id].append(img["id"])
        image_path_cache[img["id"]] = resolve_image_path(img["file_name"], args.image_dir)

    # Parse val/test/exclude objects
    val_objects = set()
    test_objects = set()
    exclude_objects = set()
    if args.val_object:
        val_objects = set(o.strip() for o in args.val_object.split(","))
    if args.test_object:
        test_objects = set(o.strip() for o in args.test_object.split(","))
    if args.exclude_object:
        exclude_objects = set(o.strip() for o in args.exclude_object.split(","))

    # Assign each image to a split
    split_image_ids = {"train": [], "valid": [], "test": []}
    missing_count = 0
    excluded_count = 0
    for img in coco["images"]:
        img_id = img["id"]
        obj_id = extract_object_id(img["file_name"])
        if obj_id is None:
            continue
        if image_path_cache.get(img_id) is None:
            missing_count += 1
            continue
        if obj_id in exclude_objects:
            excluded_count += 1
            continue
        if obj_id in test_objects:
            split_image_ids["test"].append(img_id)
        elif obj_id in val_objects:
            split_image_ids["valid"].append(img_id)
        else:
            split_image_ids["train"].append(img_id)

    if excluded_count:
        print(f"  Excluded: {excluded_count} images from {sorted(exclude_objects)}")

    if missing_count:
        print(f"  Warning: {missing_count} images not found on disk, skipping.")

    # Print split stats
    all_machines = set()
    for ids in split_image_ids.values():
        for img_id in ids:
            obj_id = extract_object_id(image_id_to_info[img_id]["file_name"])
            all_machines.add(obj_id)
    print(f"  Machines in dataset: {sorted(all_machines)}")

    for split_name in ["train", "valid", "test"]:
        ids = split_image_ids[split_name]
        machine_counts = defaultdict(int)
        for img_id in ids:
            obj_id = extract_object_id(image_id_to_info[img_id]["file_name"])
            machine_counts[obj_id] += 1
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(machine_counts.items()))
        print(f"  {split_name}: {len(ids)} images ({counts_str})")

    # Build annotation lookup
    image_id_to_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)

    # Create split directories with copied images and per-split COCO JSONs
    for split_name, img_ids in split_image_ids.items():
        split_dir = os.path.join(output_data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_images = []
        split_annotations = []
        ann_id = 1
        copied = 0
        exposure_paths = []
        skipped_no_exposure = 0

        for img_id in img_ids:
            img = image_id_to_info[img_id]
            src_path = image_path_cache.get(img_id)
            if src_path is None:
                continue

            dst_filename = os.path.basename(img["file_name"])
            dst_path = os.path.join(split_dir, dst_filename)

            # Find and copy exposure pair
            expo_src = find_exposure_pair(src_path) if args.use_exposure else None
            if args.use_exposure and expo_src is None:
                skipped_no_exposure += 1
                if args.use_exposure == "strict":
                    raise FileNotFoundError(f"No exposure pair for {src_path}")
                continue

            if not os.path.exists(dst_path) or args.force:
                shutil.copy2(src_path, dst_path)
                copied += 1

            # Apply machine-based perspective crop to bed region
            img_bgr = cv2.imread(dst_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_cropped, M = machine_based_crop(img_rgb, dst_path)
                crop_h, crop_w = img_cropped.shape[:2]
                # Save cropped image back (only if transform was applied)
                if not np.array_equal(M, np.eye(3, dtype=np.float32)):
                    cv2.imwrite(dst_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
            else:
                M = np.eye(3, dtype=np.float32)
                crop_w, crop_h = img["width"], img["height"]

            # Apply same crop to exposure pair
            if expo_src is not None:
                expo_dst = os.path.join(split_dir, "exposure", dst_filename)
                os.makedirs(os.path.dirname(expo_dst), exist_ok=True)
                if not os.path.exists(expo_dst) or args.force:
                    shutil.copy2(expo_src, expo_dst)
                expo_bgr = cv2.imread(expo_dst)
                if expo_bgr is not None:
                    expo_rgb = cv2.cvtColor(expo_bgr, cv2.COLOR_BGR2RGB)
                    expo_cropped, _ = machine_based_crop(expo_rgb, expo_dst)
                    if not np.array_equal(M, np.eye(3, dtype=np.float32)):
                        cv2.imwrite(expo_dst, cv2.cvtColor(expo_cropped, cv2.COLOR_RGB2BGR))
                exposure_paths.append(expo_dst)
            else:
                exposure_paths.append(None)

            split_images.append({
                "id": img_id,
                "width": crop_w,
                "height": crop_h,
                "file_name": dst_filename,
                "license": 0,
                "date_captured": "",
            })

            # Warp bboxes using the perspective transform
            for ann in image_id_to_annotations.get(img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id
                ann_id += 1
                if not np.array_equal(M, np.eye(3, dtype=np.float32)):
                    new_ann["bbox"] = warp_coco_bbox(
                        ann["bbox"], M, img["width"], img["height"], crop_w, crop_h
                    )
                split_annotations.append(new_ann)

        if skipped_no_exposure:
            print(f"  {split_name}: skipped {skipped_no_exposure} images without exposure pairs")

        # Save exposure paths for the custom dataset
        expo_json_path = os.path.join(split_dir, "exposure_paths.json")
        with open(expo_json_path, "w") as f:
            json.dump(exposure_paths, f)

        # Categories keep original IDs; RF-DETR remaps to 0-based internally
        split_categories = [
            {"id": cat["id"], "name": cat["name"]}
            for cat in coco["categories"]
        ]

        split_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": split_categories,
        }

        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(split_coco, f)
        print(f"  Wrote {ann_path}: {len(split_images)} images, {len(split_annotations)} annotations (copied {copied} files)")

    # Mark as prepared
    with open(skip_file, "w") as f:
        f.write(f"prepared_at={time.time()}\n")

    return output_data_dir


class PairedNormalize:
    """Normalize each 3-channel half independently for 6-channel images."""

    def __init__(self):
        self.norm = Normalize()

    def __call__(self, image, target=None):
        if image.shape[0] == 6:
            img_0, _ = self.norm(image[:3], None)
            img_1, _ = self.norm(image[3:], None)
            image = torch.cat([img_0, img_1], dim=0)
            if target is None:
                return image, None
            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                from rfdetr.util.box_ops import box_xyxy_to_cxcywh

                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target
        return self.norm(image, target)


def make_paired_transforms(image_set, resolution, aug_config, gpu_postprocess):
    """Build a transform pipeline for 6-channel paired input."""
    base = make_coco_transforms(
        image_set,
        resolution,
        multi_scale=True,
        expanded_scales=True,
        skip_random_resize=False,
        patch_size=20,
        num_windows=2,
        aug_config=aug_config,
        gpu_postprocess=gpu_postprocess,
    )
    adapted = []
    for t in base.transforms:
        if isinstance(t, AlbumentationsWrapper):
            t.return_numpy = True
            adapted.append(t)
        elif isinstance(t, Normalize):
            adapted.append(PairedNormalize())
        else:
            adapted.append(t)
    return Compose(adapted)


class PairedCocoDetection(CocoDetection):
    """CocoDetection that loads paired recoating + exposure images as 6-channel input."""

    def __init__(self, img_folder, ann_file, transforms, exposure_paths, **kwargs):
        super().__init__(img_folder, ann_file, transforms, **kwargs)
        self.exposure_paths = exposure_paths

    def _load_exposure(self, idx):
        path = self.exposure_paths[idx]
        if path is None:
            return None
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        id = self.ids[idx]
        recoat_img, raw_target = super(CocoDetection, self).__getitem__(idx)
        target = {"image_id": id, "annotations": raw_target}
        recoat_img, target = self.prepare(recoat_img, target)

        expo_img = self._load_exposure(idx)
        if expo_img is None:
            raise RuntimeError(f"Exposure image missing at index {idx} (id={id})")

        recoat_np = np.array(recoat_img)
        expo_np = np.array(expo_img)
        stacked = np.concatenate([expo_np, recoat_np], axis=2)

        if self._transforms is not None:
            img, target = self._transforms(stacked, target)
        else:
            img = stacked

        return img, target


class InputAffine(nn.Module):
    """Per-channel scale and bias applied before the patch projection conv."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


def _fix_6ch_conv(nn_model, model_config):
    """Adapt patch-embedding conv from 3ch to target_ch.
    Call AFTER load_pretrain_weights so proj.weight already has 3ch pretrained values."""
    from rfdetr.inference import _adapt_input_conv
    import copy

    patcher = nn_model.backbone[0].encoder.encoder.embeddings.patch_embeddings
    proj = patcher.projection
    target_ch = model_config.num_channels
    if proj.in_channels == target_ch:
        return

    # proj.weight already holds the 3-channel pretrained weights
    new_proj = copy.deepcopy(proj)
    new_proj.in_channels = target_ch
    new_proj.weight = torch.nn.Parameter(
        _adapt_input_conv(target_ch, proj.weight.data)
    )
    new_proj.weight.requires_grad = proj.weight.requires_grad
    patcher.num_channels = target_ch
    patcher.projection = new_proj

    # Per-channel affine before the projection conv.
    # 3-channel inputs are padded to target_ch by repeating channels.
    patcher.affine = InputAffine(target_ch)

    def _patched_forward(pixel_values):
        n_ch = pixel_values.shape[1]
        if n_ch != target_ch:
            repeats = (target_ch + n_ch - 1) // n_ch
            pixel_values = pixel_values.repeat(1, repeats, 1, 1)[:, :target_ch]
        pixel_values = patcher.affine(pixel_values)
        return patcher.projection(pixel_values).flatten(2).transpose(1, 2)

    patcher.forward = _patched_forward

    print(f"  Adapted conv projection: {proj.weight.shape} -> {new_proj.weight.shape}  (affine added)")


def _load_exposure_paths(split_dir):
    """Load exposure paths from JSON, return list or None."""
    path = os.path.join(split_dir, "exposure_paths.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


_SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}


def _build_paired_dataset(image_set, args, resolution):
    """Build a PairedCocoDetection instead of standard CocoDetection."""
    split_name = _SPLIT_FOLDER.get(image_set.split("_")[0])
    if split_name is None:
        return build_roboflow_from_coco(image_set, args, resolution)
    split_dir = os.path.join(args.dataset_dir, split_name)
    exposure_paths = _load_exposure_paths(split_dir)
    if exposure_paths is None or all(p is None for p in exposure_paths):
        return build_roboflow_from_coco(image_set, args, resolution)

    aug_config = getattr(args, "aug_config", None)
    gpu_postprocess = getattr(args, "augmentation_backend", "cpu") != "cpu"
    paired_transforms = make_paired_transforms(
        image_set, resolution, aug_config, gpu_postprocess
    )

    return PairedCocoDetection(
        img_folder=split_dir,
        ann_file=os.path.join(split_dir, "_annotations.coco.json"),
        transforms=paired_transforms,
        exposure_paths=exposure_paths,
        include_masks=False,
        remap_category_ids=True,
    )


def get_model_class(variant):
    """Import and return the appropriate RF-DETR model class."""
    import_map = {
        "2xlarge": ("rfdetr_plus", "RFDETR2XLarge"),
        "xlarge": ("rfdetr_plus", "RFDETRXLarge"),
        "large": ("rfdetr", "RFDETRLarge"),
        "medium": ("rfdetr", "RFDETRMedium"),
        "small": ("rfdetr", "RFDETRSmall"),
        "nano": ("rfdetr", "RFDETRNano"),
        "base": ("rfdetr", "RFDETRBase"),
    }
    if variant not in import_map:
        available = list(import_map.keys())
        print(f"Error: Unknown variant '{variant}'. Available: {available}")
        sys.exit(1)
    module_name, class_name = import_map[variant]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on L-PBF anomaly detection dataset"
    )

    # Data paths
    parser.add_argument("--annotation", type=str,
        default="/mnt/1tb/XAM/data/xam_first_5_folders/json_files/full_annotation.json",
        help="Path to COCO annotation JSON")
    parser.add_argument("--image-dir", type=str,
        default="/mnt/1tb/XAM/data/xam_first_5_folders/to_label",
        help="Directory containing images")
    parser.add_argument("--output", type=str,
        default="./rf-detr-output",
        help="Output directory for dataset copies and checkpoints")
    parser.add_argument("--force", action="store_true",
        help="Force re-prepare dataset even if cached")

    # Machine-ID-based split (matches existing --val-object, --test-object pattern)
    parser.add_argument("--val-object", type=str, default='SI3073',
        help="Machine ID(s) for validation, comma-separated (e.g. 'SI3781')")
    parser.add_argument("--test-object", type=str, default='SI3785',
        help="Machine ID(s) for test, comma-separated (e.g. 'SI3782')")
    parser.add_argument("--exclude-object", type=str, default='SI3397',
        help="Machine ID(s) to exclude entirely, comma-separated (e.g. 'SI3397')")

    # Model
    parser.add_argument("--variant", type=str, default="xlarge",
        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge", "base"],
        help="RF-DETR model variant")
    parser.add_argument("--resolution", type=int, default=700,
        help="Override input resolution (default: 700 for xlarge, which is the default stride-matched size)")
    parser.add_argument("--num-classes", type=int, default=7,
        help="Number of output classes")
    parser.add_argument("--num-queries", type=int, default=150,
        help="Number of object queries (default: 150)")
    parser.add_argument("--proj-size", type=int, default=0,
        help="Linear attention projection size (0 = disable, use standard attention)")
    parser.add_argument("--use-exposure", type=str, default=False, nargs="?",
        const="skip",
        help="Enable 6-channel input by pairing recoating with exposure images. "
             "'skip' skips images without pair; 'strict' raises on missing.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
        help="Number of training epochs")
    parser.add_argument("--batch-size", type=str, default="auto",
        help="Per-GPU batch size or 'auto' for automatic probing")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
        help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
        help="Number of warmup epochs (linear warmup from 0 to base LR)")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Base learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1e-5,
        help="Encoder learning rate (lower than base LR to preserve pretrained features)")
    parser.add_argument("--num-workers", type=int, default=4,
        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")
    parser.add_argument("--device", type=str, default="0",
        help="CUDA device IDs (comma-separated, e.g. '0' or '0,1')")

    # Optimization extras
    parser.add_argument("--no-ema", action="store_true",
        help="Disable Exponential Moving Average")
    parser.add_argument("--early-stopping-patience", type=int, default=30,
        help="Early stopping patience (epochs)")
    parser.add_argument("--skip-best-epochs", type=int, default=0,
        help="Skip first N epochs for early stopping best-score tracking")
    parser.add_argument("--monitor-metric", type=str, default="mAP_50",
        choices=["mAP_50", "mAP_50_95"],
        help="Validation metric to monitor for best checkpoint / early stopping")
    parser.add_argument("--no-cosine", action="store_true",
        help="Use step LR scheduler instead of cosine decay")
    parser.add_argument("--class-alpha-power", type=float, default=0.6,
        help="Power for inverse-frequency scaling of focal loss alpha (0 = uniform, 1 = fully inverse)")
    parser.add_argument("--fp16-eval", action="store_true",
        help="Use FP16 for evaluation")
    parser.add_argument("--freeze-encoder", action="store_true",
        help="Freeze the backbone encoder")
    parser.add_argument("--backbone-lora", action="store_true",
        help="Apply LoRA to the backbone")

    # Logging
    parser.add_argument("--no-wandb", action="store_true",
        help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--run-name", type=str, default=None,
        help="Custom run name for logging (default: auto-generated)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from")

    return parser.parse_args(args)


def main():
    args = parse_args()

    # Set visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Build run name
    if args.run_name:
        run_name = args.run_name
    else:
        parts = ["rfdetr", args.variant]
        if args.use_exposure:
            parts.append("6ch")
        if args.val_object:
            parts.append(f"val{args.val_object.replace(',', '_')}")
        if args.test_object:
            parts.append(f"test{args.test_object.replace(',', '_')}")
        run_name = "-".join(parts)

    print("=" * 70)
    print("RF-DETR  L-PBF  Anomaly  Detection")
    print("=" * 70)
    print(f"  Variant:      {args.variant}")
    print(f"  Run name:     {run_name}")
    print(f"  Val machines: {args.val_object or '(none)'}")
    print(f"  Test machines:{args.test_object or '(none)'}")
    print(f"  Output:       {args.output}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Device(s):    {args.device}")
    print(f"  Proj size:    {args.proj_size} {'(linear attention)' if args.proj_size > 0 else '(standard attention)'}")
    print(f"  Exposure:     {args.use_exposure or 'disabled'}")
    print("=" * 70)

    # ---- Step 1: Prepare dataset by machine-ID split ----
    print("\n[1/3] Preparing dataset (machine-ID split)...")
    t0 = time.time()
    dataset_dir = prepare_dataset(args)
    print(f"  Dataset ready at {dataset_dir} ({time.time() - t0:.1f}s)")

    # ---- Validate resolution ----
    stride = VARIANT_STRIDE[args.variant]
    default_res = VARIANT_DEFAULT_RES[args.variant]
    res = args.resolution if args.resolution is not None else default_res
    if res % stride != 0:
        print(f"Error: resolution {res} not divisible by stride {stride} "
              f"(patch_size × num_windows) for variant '{args.variant}'.")
        print(f"  Valid resolutions: {stride} × N (e.g. {max(stride, res // stride * stride)})")
        sys.exit(1)
    if res > default_res:
        pixels_ratio = (res / default_res) ** 2
        vram_est = {
            "nano": 4, "small": 6, "medium": 8, "large": 10,
            "xlarge": 12, "2xlarge": 16, "base": 6,
        }[args.variant]
        est_vram = round(vram_est * pixels_ratio, 1)
        print(f"  Warning: {res}px is {(pixels_ratio-1)*100:.0f}% more pixels than default {default_res}.")
        print(f"  Estimated VRAM for batch_size=1: ~{est_vram} GiB. Reduce --batch-size if OOM.")

    # ---- Monkey-patch for 6-channel exposure mode ----
    if args.use_exposure:
        import rfdetr.datasets as ds
        import rfdetr.training.module_model as mm

        original_build_dataset = ds.build_dataset
        def patched_build_dataset(image_set, args_, resolution):
            return _build_paired_dataset(image_set, args_, resolution)
        ds.build_dataset = patched_build_dataset

        # Monkey-patch RFDETRModelModule.__init__ to adapt conv AFTER
        # load_pretrain_weights has loaded 3ch weights. This runs only for
        # the training model (not the inference model in _build_model_context).
        original_module_init = mm.RFDETRModelModule.__init__
        def patched_module_init(self, model_config, train_config):
            original_module_init(self, model_config, train_config)
            if getattr(model_config, "num_channels", 3) != 3:
                _fix_6ch_conv(self.model, model_config)
        mm.RFDETRModelModule.__init__ = patched_module_init

        print("  6-channel exposure mode enabled")

    # ---- Step 2: Initialize model ----
    print("\n[2/3] Initializing model...")
    t0 = time.time()
    ModelClass = get_model_class(args.variant)

    model_kwargs = {
        "num_classes": args.num_classes,
        "num_queries": args.num_queries,
        "proj_size": args.proj_size,
    }
    if args.use_exposure:
        model_kwargs["num_channels"] = 6
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    if args.freeze_encoder:
        model_kwargs["freeze_encoder"] = True
    if args.backbone_lora:
        model_kwargs["backbone_lora"] = True

    # Compute per-class focal loss alpha weights from training data
    val_objects = set(o.strip() for o in args.val_object.split(",")) if args.val_object else set()
    test_objects = set(o.strip() for o in args.test_object.split(",")) if args.test_object else set()
    exclude_objects = set(o.strip() for o in args.exclude_object.split(",")) if args.exclude_object else set()
    class_alpha = compute_class_alpha(args.annotation, val_objects, test_objects, exclude_objects,
                                       power=args.class_alpha_power)
    model_kwargs["class_alpha"] = class_alpha

    print(f"  Class alpha weights: {dict(zip(CLASS_NAMES, [round(a, 3) for a in class_alpha]))}")

    model = ModelClass(**model_kwargs)
    print(f"  Model {args.variant} initialized ({time.time() - t0:.1f}s)")

    # ---- Step 3: Train ----
    print("\n[3/3] Starting training...")
    t0 = time.time()

    # Parse batch size
    batch_size_arg = args.batch_size
    if batch_size_arg != "auto":
        batch_size_arg = int(batch_size_arg)

    output_dir = os.path.join(args.output, "checkpoints", run_name)

    train_kwargs = dict(
        dataset_dir=dataset_dir,
        dataset_file="roboflow",
        epochs=args.epochs,
        batch_size=batch_size_arg,
        grad_accum_steps=args.grad_accum_steps,
        warmup_epochs=args.warmup_epochs,
        output_dir=output_dir,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        project="l-pbf-anomaly-detection",
        run=run_name,
        class_names=CLASS_NAMES,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        skip_best_epochs=args.skip_best_epochs,
        monitor_metric=args.monitor_metric,
        lr_scheduler="step" if args.no_cosine else "cosine",
        num_workers=args.num_workers,
        seed=args.seed,
        tensorboard=True,
        wandb=not args.no_wandb,
        use_ema=not args.no_ema,
        fp16_eval=args.fp16_eval,
        aug_config={
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.5},
            "RandomBrightnessContrast": {"brightness_limit": 0.25, "contrast_limit": 0.25, "p": 0.6},
            "CLAHE": {"clip_limit": 4.0, "tile_grid_size": (8, 8), "p": 0.5},
            "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},
            "Sharpen": {"alpha": (0.2, 0.5), "lightness": (0.5, 1.0), "p": 0.3},
            "GaussNoise": {"p": 0.2},
            "GaussianBlur": {"blur_limit": 3, "p": 0.15},
        },
    )

    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)

    print(f"\n Training complete! ({time.time() - t0:.1f}s)")
    print(f"  Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()

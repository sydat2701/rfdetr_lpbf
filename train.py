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

    # Parse val/test objects
    val_objects = set()
    test_objects = set()
    if args.val_object:
        val_objects = set(o.strip() for o in args.val_object.split(","))
    if args.test_object:
        test_objects = set(o.strip() for o in args.test_object.split(","))

    # Assign each image to a split
    split_image_ids = {"train": [], "valid": [], "test": []}
    missing_count = 0
    for img in coco["images"]:
        img_id = img["id"]
        obj_id = extract_object_id(img["file_name"])
        if obj_id is None:
            continue
        if image_path_cache.get(img_id) is None:
            missing_count += 1
            continue
        if obj_id in test_objects:
            split_image_ids["test"].append(img_id)
        elif obj_id in val_objects:
            split_image_ids["valid"].append(img_id)
        else:
            split_image_ids["train"].append(img_id)

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

        for img_id in img_ids:
            img = image_id_to_info[img_id]
            src_path = image_path_cache.get(img_id)
            if src_path is None:
                continue

            dst_filename = os.path.basename(img["file_name"])
            dst_path = os.path.join(split_dir, dst_filename)

            if not os.path.exists(dst_path) or args.force:
                shutil.copy2(src_path, dst_path)
                copied += 1

            split_images.append({
                "id": img_id,
                "width": img["width"],
                "height": img["height"],
                "file_name": dst_filename,
                "license": 0,
                "date_captured": "",
            })

            for ann in image_id_to_annotations.get(img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id
                ann_id += 1
                split_annotations.append(new_ann)

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
    parser.add_argument("--val-object", type=str, default='SI3781',
        help="Machine ID(s) for validation, comma-separated (e.g. 'SI3781')")
    parser.add_argument("--test-object", type=str, default=None,
        help="Machine ID(s) for test, comma-separated (e.g. 'SI3782')")

    # Model
    parser.add_argument("--variant", type=str, default="large",
        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge", "base"],
        help="RF-DETR model variant")
    parser.add_argument("--resolution", type=int, default=928,
        help="Override input resolution (default: variant-specific)")
    parser.add_argument("--num-classes", type=int, default=7,
        help="Number of output classes")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
        help="Number of training epochs")
    parser.add_argument("--batch-size", type=str, default="2",
        help="Per-GPU batch size or 'auto' for automatic probing")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Base learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4,
        help="Encoder learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")
    parser.add_argument("--device", type=str, default="0",
        help="CUDA device IDs (comma-separated, e.g. '0' or '0,1')")

    # Optimization extras
    parser.add_argument("--no-ema", action="store_true",
        help="Disable Exponential Moving Average")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
        help="Early stopping patience (epochs)")
    parser.add_argument("--fp16-eval", action="store_true",
        help="Use FP16 for evaluation")
    parser.add_argument("--freeze-encoder", action="store_true",
        help="Freeze the backbone encoder")
    parser.add_argument("--backbone-lora", action="store_true",
        help="Apply LoRA to the backbone")

    # Logging
    parser.add_argument("--wandb", action="store_true",
        help="Enable Weights & Biases logging")
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
        if args.val_object:
            parts.append(f"val{args.val_object.replace(',', '_')}")
        if args.test_object:
            parts.append(f"test{args.test_object.replace(',', '_')}")
        run_name = "-".join(parts)

    print("=" * 80)
    print("RF-DETR  L-PBF  Anomaly  Detection")
    print("=" * 80)
    print(f"  Variant:      {args.variant}")
    print(f"  Run name:     {run_name}")
    print(f"  Val machines: {args.val_object or '(none)'}")
    print(f"  Test machines:{args.test_object or '(none)'}")
    print(f"  Output:       {args.output}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Device(s):    {args.device}")
    print("=" * 80)

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

    # ---- Step 2: Initialize model ----
    print("\n[2/3] Initializing model...")
    t0 = time.time()
    ModelClass = get_model_class(args.variant)

    model_kwargs = {
        "num_classes": args.num_classes,
    }
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    if args.freeze_encoder:
        model_kwargs["freeze_encoder"] = True
    if args.backbone_lora:
        model_kwargs["backbone_lora"] = True

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
        output_dir=output_dir,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        project="l-pbf-anomaly-detection",
        run=run_name,
        class_names=CLASS_NAMES,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        tensorboard=True,
        wandb=args.wandb,
        use_ema=not args.no_ema,
        fp16_eval=args.fp16_eval,
        aug_config={
            "HorizontalFlip": {"p": 0.3},
            "VerticalFlip": {"p": 0.3},
            "Rotate": {"limit": (-2, 2), "border_mode": 4, "p": 0.1},
            "RandomScale": {"scale_limit": 0.005, "p": 0.1},
            "RandomGamma": {"gamma_limit": (80, 120), "p": 0.1},
            "Sharpen": {"alpha": (0.1, 0.4), "lightness": (1.0, 1.0), "p": 0.1},
            "RandomBrightnessContrast": {
                "brightness_limit": 0.15,
                "contrast_limit": 0.15,
                "p": 0.3,
            },
        },
    )

    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)

    print(f"\n Training complete! ({time.time() - t0:.1f}s)")
    print(f"  Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()

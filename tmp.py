#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from collections import Counter, defaultdict


AUG_LOW_CONTRAST_SMALL_DEFECT = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.25, "contrast_limit": 0.25, "p": 0.6},
    "CLAHE": {"clip_limit": 4.0, "tile_grid_size": (8, 8), "p": 0.5},
    "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},
    "Sharpen": {"alpha": (0.2, 0.5), "lightness": (0.5, 1.0), "p": 0.3},
    "GaussNoise": {"p": 0.2},
    "ISONoise": {"p": 0.2},
}

# =============================================================================
# 2. CẤU HÌNH TRAINING MẶC ĐỊNH
# =============================================================================
DEFAULTS = dict(
    model="medium",        # nano|small|medium|base|large  (medium: cân bằng tốt độ/độ chính xác)
    resolution=896,        # Bội số hợp lệ KHÁC NHAU tuỳ model (16*num_windows).
                           #   896 = 32*28 = 56*16 -> AN TOÀN cho mọi biến thể.
                           #   Lỗi nhỏ -> để CAO: thử 1024 (model chia hết 32) hoặc
                           #   1120 (an toàn mọi model) nếu GPU đủ VRAM.
    epochs=80,
    batch_size=4,          # mini-batch mỗi GPU; giảm còn 2 nếu hết VRAM
    grad_accum_steps=4,    # effective batch = batch_size * grad_accum_steps * num_gpus = 16
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005,
    skip_best_epochs=5,
    use_ema=True,          
    num_workers=4,
    gradient_checkpointing=False,
)

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}


# =============================================================================
# BƯỚC 1: KIỂM TRA DATASET
# =============================================================================
def step_check(dataset_dir):
    print("=" * 70)
    print("BƯỚC 1 — KIỂM TRA DATASET")
    print("=" * 70)
    ok = True
    cat_names = None
    for split in ["train", "valid", "test"]:
        jpath = os.path.join(dataset_dir, split, "_annotations.coco.json")
        if not os.path.isfile(jpath):
            print(f"  [THIẾU] {jpath}")
            ok = False
            continue
        d = json.load(open(jpath, encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        if cat_names is None:
            cat_names = cats
        dist = Counter(a["category_id"] for a in d["annotations"])
        ncls = sum(1 for cid in cats if dist.get(cid, 0) > 0)
        # đếm ảnh thực sự tồn tại trên đĩa
        img_dir = os.path.join(dataset_dir, split)
        missing = sum(1 for im in d["images"]
                      if not os.path.isfile(os.path.join(img_dir, im["file_name"])))
        print(f"  [{split:<5}] ảnh={len(d['images']):>5}  ann={len(d['annotations']):>7}  "
              f"class={ncls}/{len(cats)}  ảnh-file-thiếu={missing}")
        per = "  ".join(f"{cats[c][:8]}={dist.get(c,0)}" for c in sorted(cats))
        print(f"          {per}")
        if missing:
            print(f"          !! Có {missing} ảnh trong JSON nhưng KHÔNG có file -> sẽ lỗi khi train.")
            ok = False
    print()
    if ok:
        print(">> Dataset OK, sẵn sàng train.")
    else:
        print(">> Dataset CÓ VẤN ĐỀ, hãy xử lý trước khi train.")
    return ok


# =============================================================================
# BƯỚC 2: TRAINING
# =============================================================================
def step_train(args):
    print("=" * 70)
    print("BƯỚC 2 — TRAINING RF-DETR")
    print("=" * 70)
    import rfdetr
    ModelClass = getattr(rfdetr, MODEL_CLASSES[args.model])

    if args.resolution % 32 != 0 and args.resolution % 56 != 0:
        print(f"  [!] resolution={args.resolution} có thể không hợp lệ. "
              f"Gợi ý an toàn (chia hết cả 32 & 56): 896, 1120. "
              f"Hoặc theo model: bội số của 32 (vd 1024) hay 56 (vd 1008).")

    # --- Preset tiết kiệm VRAM (cho model lớn / GPU nhỏ như 16GB) ---
    if args.low_vram:
        print("  [LOW-VRAM] Ép batch_size=1, grad_accum=16, gradient_checkpointing=ON, EMA=OFF")
        args.batch_size = 1
        args.grad_accum_steps = 16
        args.gradient_checkpointing = True
        args.use_ema = False

    if args.model in ("xlarge", "2xlarge"):
        print("  [!] XLarge/2XLarge cần: pip install rfdetr_plus  (giấy phép PML 1.0).")
        print("      Trên GPU 16GB, model này rất dễ OOM kể cả ở resolution thấp.")
        print("      Cân nhắc --model large --resolution 1008 thường tốt hơn cho lỗi nhỏ.")

    # Khởi tạo model. Nếu có file .pth riêng -> nạp làm trọng số khởi đầu.
    init_kwargs = dict(resolution=args.resolution)
    if args.pretrain_weights:
        init_kwargs["pretrain_weights"] = args.pretrain_weights
        print(f"  Nạp trọng số khởi đầu: {args.pretrain_weights}")
    model = ModelClass(**init_kwargs)

    # Chọn augmentation
    if args.aug == "custom":
        aug_config = AUG_LOW_CONTRAST_SMALL_DEFECT
    elif args.aug == "industrial":
        from rfdetr.datasets.aug_config import AUG_INDUSTRIAL
        aug_config = AUG_INDUSTRIAL
    elif args.aug == "conservative":
        from rfdetr.datasets.aug_config import AUG_CONSERVATIVE
        aug_config = AUG_CONSERVATIVE
    else:  # "default" -> chỉ HorizontalFlip 50% của RF-DETR
        aug_config = None

    print(f"  Model        : {MODEL_CLASSES[args.model]}")
    print(f"  Resolution   : {args.resolution}")
    print(f"  Epochs       : {args.epochs}  | batch={args.batch_size} x accum={args.grad_accum_steps}"
          f" (effective={args.batch_size*args.grad_accum_steps})")
    print(f"  Augmentation : {args.aug}")
    print(f"  Output       : {args.output_dir}")
    print()

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        resolution=args.resolution,
        num_workers=args.num_workers,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        skip_best_epochs=args.skip_best_epochs,
        use_ema=args.use_ema,
        gradient_checkpointing=args.gradient_checkpointing,
        tensorboard=True,   # xem log: tensorboard --logdir <output_dir>
    )
    if aug_config is not None:
        train_kwargs["aug_config"] = aug_config
    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)
    print("\n>> Train xong. Trọng số tốt nhất: ",
          os.path.join(args.output_dir, "checkpoint_best_total.pth"))


# =============================================================================
# BƯỚC 3: ĐÁNH GIÁ THEO TỪNG CLASS TRÊN TẬP TEST
# =============================================================================
def step_eval(args):
    print("=" * 70)
    print("BƯỚC 3 — ĐÁNH GIÁ TRÊN TẬP TEST (AP từng class)")
    print("=" * 70)
    import numpy as np
    from PIL import Image
    import rfdetr
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ckpt = args.weights or os.path.join(args.output_dir, "checkpoint_best_total.pth")
    if not os.path.isfile(ckpt):
        sys.exit(f"Không thấy checkpoint: {ckpt}")
    ModelClass = getattr(rfdetr, MODEL_CLASSES[args.model])
    model = ModelClass(pretrain_weights=ckpt, resolution=args.resolution)

    test_dir = os.path.join(args.dataset_dir, "test")
    gt_json = os.path.join(test_dir, "_annotations.coco.json")
    coco_gt = COCO(gt_json)
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    # Chạy dự đoán, gom kết quả theo định dạng COCO detection.
    results = []
    img_ids = coco_gt.getImgIds()
    for k, img_id in enumerate(img_ids, 1):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(test_dir, info["file_name"])
        image = Image.open(img_path).convert("RGB")
        det = model.predict(image, threshold=args.conf)  # supervision Detections
        if det.xyxy is None or len(det.xyxy) == 0:
            continue
        for (x1, y1, x2, y2), score, cls in zip(det.xyxy, det.confidence, det.class_id):
            results.append({
                "image_id": img_id,
                # LƯU Ý: với COCO 1-indexed như dữ liệu này, class_id dự đoán thường
                # trùng category_id. Nếu AP=0 bất thường, kiểm tra lại ánh xạ id ở đây.
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
        if k % 50 == 0:
            print(f"    ...đã dự đoán {k}/{len(img_ids)} ảnh")

    if not results:
        sys.exit("Mô hình không dự đoán được box nào (thử giảm --conf).")

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    # AP theo từng class (mAP@[.5:.95] cho mỗi category)
    print("\n=== AP TỪNG CLASS (IoU .50:.95) ===")
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    cat_ids = coco_gt.getCatIds()
    for idx, cid in enumerate(cat_ids):
        p = precisions[:, :, idx, 0, -1]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size else float("nan")
        print(f"  {cats[cid]:<16}: AP = {ap:.4f}")
    print("\n>> Nhìn AP từng class: class hiếm (shortage/hole/powder residue) thường thấp,")
    print("   đó là nơi cần thêm dữ liệu hoặc augment/oversample mạnh hơn.")


# =============================================================================
# BƯỚC 4: EXPORT ONNX
# =============================================================================
def step_export(args):
    print("=" * 70)
    print("BƯỚC 4 — EXPORT ONNX")
    print("=" * 70)
    import rfdetr
    ckpt = args.weights or os.path.join(args.output_dir, "checkpoint_best_total.pth")
    ModelClass = getattr(rfdetr, MODEL_CLASSES[args.model])
    model = ModelClass(pretrain_weights=ckpt, resolution=args.resolution)
    model.export()  # tạo file ONNX trong output (cần: pip install "rfdetr[onnxexport]")
    print(">> Đã export ONNX (xem thư mục output).")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Pipeline RF-DETR cho phát hiện lỗi bề mặt.")
    ap.add_argument("--dataset-dir", required=True, help="Thư mục dataset (chứa train/valid/test).")
    ap.add_argument("--output-dir", default="output_rfdetr", help="Thư mục lưu log + checkpoint.")
    ap.add_argument("--steps", nargs="+", default=["check"],
                    choices=["check", "train", "eval", "export"],
                    help="Các bước cần chạy, ví dụ: --steps check train eval export")
    # model / training
    ap.add_argument("--model", default=DEFAULTS["model"], choices=list(MODEL_CLASSES))
    ap.add_argument("--resolution", type=int, default=DEFAULTS["resolution"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum_steps"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--lr-encoder", type=float, default=DEFAULTS["lr_encoder"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                    default=DEFAULTS["early_stopping"])
    ap.add_argument("--early-stopping-patience", type=int, default=DEFAULTS["early_stopping_patience"])
    ap.add_argument("--early-stopping-min-delta", type=float, default=DEFAULTS["early_stopping_min_delta"])
    ap.add_argument("--skip-best-epochs", type=int, default=DEFAULTS["skip_best_epochs"])
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=DEFAULTS["use_ema"])
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    default=DEFAULTS["gradient_checkpointing"])
    ap.add_argument("--low-vram", action="store_true",
                    help="Preset GPU nhỏ: batch=1, accum=16, gradient_checkpointing, tắt EMA.")
    ap.add_argument("--pretrain-weights", default=None,
                    help="Đường dẫn .pth để nạp làm trọng số khởi đầu (vd file 2xlarge của bạn).")
    ap.add_argument("--aug", default="custom",
                    choices=["custom", "industrial", "conservative", "default"],
                    help="custom=cấu hình tối ưu ảnh khó nhìn; industrial=preset công nghiệp.")
    ap.add_argument("--resume", default=None, help="Đường dẫn last.ckpt để học tiếp.")
    # eval / export
    ap.add_argument("--weights", default=None, help="Checkpoint dùng cho eval/export.")
    ap.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy khi đánh giá.")
    args = ap.parse_args()

    if "check" in args.steps:
        step_check(args.dataset_dir)
    if "train" in args.steps:
        step_train(args)
    if "eval" in args.steps:
        step_eval(args)
    if "export" in args.steps:
        step_export(args)


if __name__ == "__main__":
    main()

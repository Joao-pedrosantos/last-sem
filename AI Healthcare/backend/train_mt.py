"""
Multi-task training: pneumonia classification + weakly-supervised localization.

We supervise the Class Activation Map (CAM) directly with the bounding-box
ground truth from `stage_2_train_labels.csv`. This forces the model to
*localize* inside the bbox rather than learn whatever global feature
correlates with the label.

Key idea: the existing PneumoniaClassifier ends in `GAP → Linear`. That is
mathematically identical to `Linear-as-1x1-conv → GAP`. By computing the CAM
explicitly we can supervise it spatially — same parameters, no new weights.
The state_dict stays compatible with the original classifier so we warm-start
from `outputs/best_model.pt` (already a decent classifier).

Loss   = FocalLoss(logit, label)               # classification signal
       + λ * BCE(CAM_upsampled, bbox_mask)     # localization signal

For negatives (label=0) bbox_mask is all-zeros, which pushes the CAM down
everywhere — also a useful signal.

Usage:
    python backend/train_mt.py
    python backend/train_mt.py --epochs 10 --lambda-seg 0.5
"""
from __future__ import annotations

# Pre-load torchxrayvision.baseline_models so jfhealthcare's broken
# `from model.utils import get_norm` resolves to its own bundled package
# (torchxrayvision hacks sys.path to make it work). Then we *undo* that
# pollution so our own backend/model.py is importable below — without this
# cleanup `from model import PneumoniaClassifier` picks up jfhealthcare's
# `model` package instead.
import sys as _sys
import torchxrayvision.baseline_models  # noqa: F401, E402
_sys.modules.pop("model", None)
_sys.path[:] = [p for p in _sys.path
                if "jfhealthcare" not in p and "chexpert" not in p]

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cxr_pipeline import (  # noqa: E402
    ImageNetMean, ImageNetStd, lung_mask as compute_lung_mask, read_dicom,
    _pad_to_square, _pad_to_square_fill,
)
from model import PneumoniaClassifier, PneumoniaClassifierMT  # noqa: E402
from training import FocalLoss, threshold_at_specificity  # noqa: E402

SEED = 42
DATA_DIR = ROOT / "rsna-pneumonia-detection-challenge"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "preproc_cache_mt"
# Prefer the MT-trained checkpoint if it exists (resuming a partial MT run),
# otherwise warm-start from the single-task best — both have identical
# state_dict layout so either loads cleanly.
WARM_START_CKPT_MT = OUT_DIR / "best_model_mt.pt"
WARM_START_CKPT_ST = OUT_DIR / "best_model.pt"


# ---------------------------------------------------------------------------
# Multi-task model — shares all parameters with PneumoniaClassifier
# ---------------------------------------------------------------------------
# (PneumoniaClassifierMT now lives in backend/model.py — imported above.)


# ---------------------------------------------------------------------------
# Cropping that keeps the bbox mask aligned with the image
# ---------------------------------------------------------------------------
def lung_focused_crop_with_bbox(
    img_gray: np.ndarray, bbox_mask: np.ndarray, pad_frac: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same crop policy as `cxr_pipeline.lung_focused_crop`, but the bbox mask is
    cropped + padded with the *exact same* coordinates so it stays aligned.

    Returns (cropped_img, lung_mask_crop, bbox_mask_crop) all at the same H×W.
    """
    assert img_gray.shape == bbox_mask.shape
    lung = compute_lung_mask(img_gray)
    if int(lung.sum()) < 1000:
        # Lung segmentation failed — fall back to identity crop, mask is zeros.
        sq = _pad_to_square(img_gray)
        return sq, np.ones_like(sq, dtype=np.uint8), _pad_to_square(bbox_mask)

    ys, xs = np.where(lung > 0)
    h, w = img_gray.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad_y = int(round((y1 - y0) * pad_frac))
    pad_x = int(round((x1 - x0) * pad_frac))
    y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y + 1)
    x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x + 1)

    img_crop = img_gray[y0:y1, x0:x1]
    lung_crop = lung[y0:y1, x0:x1]
    bbox_crop = bbox_mask[y0:y1, x0:x1]

    fill = int(img_crop[lung_crop > 0].mean()) if lung_crop.any() else 128
    img_masked = np.where(lung_crop > 0, img_crop, fill).astype(np.uint8)

    img_sq = _pad_to_square_fill(img_masked, fill)
    lung_sq = _pad_to_square(lung_crop)
    bbox_sq = _pad_to_square(bbox_crop)
    return img_sq, lung_sq, bbox_sq


# ---------------------------------------------------------------------------
# Dataset — caches (image, bbox_mask) per patient to disk
# ---------------------------------------------------------------------------
def build_bbox_lookup(labels_csv: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Return (patient_df, bbox_dict).

    patient_df: one row per patientId with column `target` (0/1).
    bbox_dict:  patientId -> list of (x, y, w, h) tuples (empty for negatives).
    """
    raw = pd.read_csv(labels_csv)
    patient_df = raw.groupby("patientId", as_index=False).agg(target=("Target", "max"))

    bbox_dict: dict[str, list[tuple[float, float, float, float]]] = {}
    for pid, group in raw.groupby("patientId"):
        boxes = []
        for _, r in group.iterrows():
            if pd.notna(r["x"]):
                boxes.append((float(r["x"]), float(r["y"]),
                              float(r["width"]), float(r["height"])))
        bbox_dict[pid] = boxes
    return patient_df, bbox_dict


def render_bbox_mask(shape, boxes) -> np.ndarray:
    """Binary uint8 mask: 1 inside any bbox, 0 elsewhere."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
        x1 = min(w, int(round(x + bw))); y1 = min(h, int(round(y + bh)))
        mask[y0:y1, x0:x1] = 1
    return mask


def make_train_transforms_mt(size: int) -> A.Compose:
    """Augmentations applied to image + bbox_mask together (albumentations syncs)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                           rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # No CoarseDropout — it could blank parts of the bbox region and mess
        # with the localization loss. Lung masking already suppresses border
        # shortcuts.
        A.Normalize(mean=ImageNetMean, std=ImageNetStd),
        ToTensorV2(),
    ])


def make_val_transforms_mt() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=ImageNetMean, std=ImageNetStd),
        ToTensorV2(),
    ])


class RSNADatasetMT(Dataset):
    def __init__(self, df: pd.DataFrame, bbox_dict: dict, img_dir: Path,
                 transform: A.Compose, img_size: int):
        self.df = df.reset_index(drop=True)
        self.bbox_dict = bbox_dict
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_or_compute(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{pid}.npz"
        if cache.exists():
            d = np.load(cache)
            return d["img"], d["bbox_mask"]
        img_gray = read_dicom(self.img_dir / f"{pid}.dcm")
        bbox_mask_orig = render_bbox_mask(img_gray.shape, self.bbox_dict.get(pid, []))
        img, _, bbox_mask = lung_focused_crop_with_bbox(img_gray, bbox_mask_orig)
        np.savez_compressed(cache, img=img, bbox_mask=bbox_mask)
        return img, bbox_mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row["patientId"])
        target = float(row["target"])
        img, bbox_mask = self._load_or_compute(pid)

        # Resize both to model input size, then transform.
        pil = Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(pil)
        rgb = np.stack([arr, arr, arr], axis=-1)

        # Mask: nearest-neighbour so it stays binary.
        bbox_pil = Image.fromarray(bbox_mask * 255).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        bbox_arr = (np.array(bbox_pil) > 127).astype(np.float32)

        out = self.transform(image=rgb, mask=bbox_arr)
        return out["image"], out["mask"], torch.tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def freeze_for_finetune(model: PneumoniaClassifierMT, n_unfreeze: int = 2) -> None:
    """Freeze everything except the last `n_unfreeze` MBConv blocks and head."""
    for p in model.model.parameters():
        p.requires_grad = False
    blocks = model.model.blocks
    for i in range(max(0, len(blocks) - n_unfreeze), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    for p in model.model.get_classifier().parameters():
        p.requires_grad = True
    # conv_head + bn2 are also small and benefit from fine-tune
    if hasattr(model.model, "conv_head"):
        for p in model.model.conv_head.parameters():
            p.requires_grad = True
    if hasattr(model.model, "bn2"):
        for p in model.model.bn2.parameters():
            p.requires_grad = True


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_one_epoch(model, loader, focal, bce, optimizer, device, scaler,
                    lambda_seg: float, img_size: int):
    model.train()
    cls_total, seg_total, n = 0.0, 0.0, 0
    for imgs, bbox_masks, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        bbox_masks = bbox_masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logit, cam = model(imgs, return_cam=True)
            cam_up = F.interpolate(cam, size=(img_size, img_size),
                                   mode="bilinear", align_corners=False).squeeze(1)
            cls_loss = focal(logit, targets)
            seg_loss = bce(cam_up, bbox_masks)
            loss = cls_loss + lambda_seg * seg_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = imgs.size(0)
        cls_total += cls_loss.item() * bs
        seg_total += seg_loss.item() * bs
        n += bs
    return cls_total / n, seg_total / n


@torch.no_grad()
def evaluate(model, loader, focal, bce, device, lambda_seg: float, img_size: int):
    model.eval()
    cls_total, seg_total, n = 0.0, 0.0, 0
    probs_all, labels_all = [], []
    for imgs, bbox_masks, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        bbox_masks = bbox_masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logit, cam = model(imgs, return_cam=True)
        cam_up = F.interpolate(cam, size=(img_size, img_size),
                               mode="bilinear", align_corners=False).squeeze(1)
        cls_loss = focal(logit, targets)
        seg_loss = bce(cam_up, bbox_masks)
        bs = imgs.size(0)
        cls_total += cls_loss.item() * bs
        seg_total += seg_loss.item() * bs
        n += bs
        probs_all.append(torch.sigmoid(logit).cpu().numpy())
        labels_all.append(targets.cpu().numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    auc = roc_auc_score(labels, probs)
    return cls_total / n, seg_total / n, auc, probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda-seg", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-warm-start", action="store_true",
                    help="skip loading outputs/best_model.pt as init")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}", flush=True)

    patient_df, bbox_dict = build_bbox_lookup(LABELS_CSV)
    print(f"Patients: {len(patient_df):,} | positives: {patient_df.target.sum():,}",
          flush=True)

    train_df, temp_df = train_test_split(
        patient_df, test_size=0.2, stratify=patient_df["target"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["target"], random_state=SEED
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}",
          flush=True)

    train_tf = make_train_transforms_mt(args.img_size)
    val_tf = make_val_transforms_mt()
    train_ds = RSNADatasetMT(train_df, bbox_dict, TRAIN_IMG_DIR, train_tf, args.img_size)
    val_ds = RSNADatasetMT(val_df, bbox_dict, TRAIN_IMG_DIR, val_tf, args.img_size)
    test_ds = RSNADatasetMT(test_df, bbox_dict, TRAIN_IMG_DIR, val_tf, args.img_size)

    # persistent_workers=False — Windows DataLoaders can deadlock when workers
    # span many epochs; let them respawn cleanly each epoch instead.
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = PneumoniaClassifierMT("efficientnet_b4", pretrained=True, dropout=0.3).to(device)
    if not args.no_warm_start:
        ckpt = WARM_START_CKPT_MT if WARM_START_CKPT_MT.exists() else WARM_START_CKPT_ST
        if ckpt.exists():
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Warm-start loaded from {ckpt.name} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    freeze_for_finetune(model, n_unfreeze=2)
    print(f"Trainable: {count_trainable(model)/1e6:.2f}M", flush=True)

    focal = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05).to(device)
    bce = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler(device.type)

    best_auc = 0.0
    best_ckpt = OUT_DIR / "best_model_mt.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_cls, tr_seg = train_one_epoch(
            model, train_loader, focal, bce, optimizer, device, scaler,
            args.lambda_seg, args.img_size,
        )
        va_cls, va_seg, va_auc, _, _ = evaluate(
            model, val_loader, focal, bce, device, args.lambda_seg, args.img_size,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        flag = ""
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_ckpt)
            flag = " *"
        line = (
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"TL_cls {tr_cls:.4f} TL_seg {tr_seg:.4f} | "
            f"VL_cls {va_cls:.4f} VL_seg {va_seg:.4f} | "
            f"VAUC {va_auc:.4f} | LR {lr:.2e} | {time.time()-t0:.0f}s{flag}"
        )
        print(line, flush=True)
        history.append({
            "epoch": epoch, "train_cls": tr_cls, "train_seg": tr_seg,
            "val_cls": va_cls, "val_seg": va_seg, "val_auc": va_auc, "lr": lr,
        })

    # --- Final eval on best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, _, val_auc, val_probs, val_labels = evaluate(
        model, val_loader, focal, bce, device, args.lambda_seg, args.img_size,
    )
    _, _, test_auc, _, _ = evaluate(
        model, test_loader, focal, bce, device, args.lambda_seg, args.img_size,
    )
    threshold = threshold_at_specificity(val_labels, val_probs, target_spec=0.95)
    print(f"\nFinal Val AUC:  {val_auc:.4f}", flush=True)
    print(f"Final Test AUC: {test_auc:.4f}", flush=True)
    print(f"Clinical threshold (val Spec=0.95): {threshold:.4f}", flush=True)

    final_ckpt = OUT_DIR / "pneumonia_model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": args.img_size,
        "threshold": float(threshold),
        "model_name": "efficientnet_b4_mt",
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "lambda_seg": float(args.lambda_seg),
        "multi_task": True,
    }, final_ckpt)
    with (OUT_DIR / "training_mt_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved: {final_ckpt}", flush=True)


if __name__ == "__main__":
    main()

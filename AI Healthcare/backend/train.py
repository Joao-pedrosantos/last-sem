"""
Standalone training script for the CXR pneumonia classifier.

Replaces the training cells in `project.ipynb`. Identical preprocessing to
the serving backend (lung-segmented crop via `cxr_pipeline.lung_focused_crop`)
so train/serve drift is impossible by construction.

Outputs `outputs/pneumonia_model_final.pt` with `img_size` and `threshold`
metadata embedded; backend/main.py reads those keys at boot.

Usage:
    python backend/train.py                     # defaults
    python backend/train.py --epochs-phase2 12  # longer fine-tune

First epoch is slow because every image is segmented and cached to disk under
`outputs/preproc_cache/`. Subsequent epochs read the cached crops directly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cxr_pipeline import (  # noqa: E402
    lung_focused_crop,
    make_train_transforms,
    make_val_transforms,
    read_dicom,
)
from model import PneumoniaClassifier  # noqa: E402
from training import FocalLoss, threshold_at_specificity  # noqa: E402

SEED = 42
DATA_DIR = ROOT / "rsna-pneumonia-detection-challenge"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "preproc_cache"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RSNADataset(Dataset):
    """
    Lung-segmented RSNA pneumonia dataset.

    On first access, reads the DICOM, runs lung segmentation, crops + masks,
    and caches a (crop, mask) pair as two .npy files keyed by patientId. Later
    epochs only do the cheap resize + augmentation work.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_or_compute(self, pid: str) -> np.ndarray:
        cache_img = CACHE_DIR / f"{pid}.npy"
        if cache_img.exists():
            return np.load(cache_img)
        dcm_path = self.img_dir / f"{pid}.dcm"
        img_gray = read_dicom(dcm_path)
        cropped, _ = lung_focused_crop(img_gray)
        np.save(cache_img, cropped)
        return cropped

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row["patientId"])
        target = float(row["target"])

        cropped = self._load_or_compute(pid)
        pil = Image.fromarray(cropped).resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(pil)
        rgb = np.stack([arr, arr, arr], axis=-1)

        out = self.transform(image=rgb)
        return out["image"], torch.tensor(target, dtype=torch.float32)


def build_dataframe() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    # Multiple bbox rows per patient → reduce to one binary label per patient.
    return df.groupby("patientId", as_index=False).agg(target=("Target", "max"))


def freeze_backbone(model: PneumoniaClassifier) -> None:
    for p in model.model.parameters():
        p.requires_grad = False
    head = model.model.get_classifier()
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: PneumoniaClassifier, n: int = 2) -> None:
    """EfficientNet-B4 timm layout: model.blocks is the sequential MBConv stack."""
    blocks = model.model.blocks
    for p in model.model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks) - n), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    head = model.model.get_classifier()
    for p in head.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(imgs).view(-1)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    probs_all, labels_all = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs).view(-1)
        loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(targets.cpu().numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    auc = roc_auc_score(labels, probs)
    return total_loss / max(n, 1), auc, probs, labels


def run_phase(name, model, train_loader, val_loader, criterion, optimizer,
              scheduler, device, scaler, epochs, best_auc, best_ckpt):
    print(f"\n=== {name} === ({count_trainable(model)/1e6:.2f}M trainable params)")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl, vauc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        flag = ""
        if vauc > best_auc:
            best_auc = vauc
            torch.save(model.state_dict(), best_ckpt)
            flag = " *"
        print(
            f"  Epoch {epoch:2d}/{epochs} | TL {tl:.4f} | VL {vl:.4f} | "
            f"VAUC {vauc:.4f} | LR {lr:.2e} | {time.time()-t0:.0f}s{flag}"
        )
    return best_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-phase1", type=int, default=5)
    ap.add_argument("--epochs-phase2", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--lr-phase1", type=float, default=1e-4)
    ap.add_argument("--lr-phase2", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    df = build_dataframe()
    print(f"Total samples: {len(df):,} (positive rate {df['target'].mean():.3f})")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["target"], random_state=SEED
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_tf = make_train_transforms(args.img_size)
    val_tf = make_val_transforms(args.img_size)
    train_ds = RSNADataset(train_df, TRAIN_IMG_DIR, train_tf, args.img_size)
    val_ds = RSNADataset(val_df, TRAIN_IMG_DIR, val_tf, args.img_size)
    test_ds = RSNADataset(test_df, TRAIN_IMG_DIR, val_tf, args.img_size)

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = PneumoniaClassifier("efficientnet_b4", pretrained=True, dropout=0.3).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05).to(device)
    scaler = torch.amp.GradScaler(device.type)

    best_auc = 0.0
    best_ckpt = OUT_DIR / "best_model.pt"

    # --- Phase 1: classifier head only
    freeze_backbone(model)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr_phase1)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs_phase1, eta_min=args.lr_phase1 * 0.1
    )
    best_auc = run_phase(
        "Phase 1 — frozen backbone", model, train_loader, val_loader,
        criterion, opt, sched, device, scaler, args.epochs_phase1,
        best_auc, best_ckpt,
    )

    # --- Phase 2: unfreeze last 2 MBConv stages
    unfreeze_last_blocks(model, n=2)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr_phase2)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.1
    )
    best_auc = run_phase(
        "Phase 2 — fine-tune last blocks", model, train_loader, val_loader,
        criterion, opt, sched, device, scaler, args.epochs_phase2,
        best_auc, best_ckpt,
    )

    # --- Final eval on best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, val_auc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    _, test_auc, _, _ = evaluate(model, test_loader, criterion, device)
    threshold = threshold_at_specificity(val_labels, val_probs, target_spec=0.95)
    print(f"\nFinal Val AUC:  {val_auc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Clinical threshold (val Spec=0.95): {threshold:.4f}")

    final_ckpt = OUT_DIR / "pneumonia_model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": args.img_size,
        "threshold": float(threshold),
        "model_name": "efficientnet_b4",
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
    }, final_ckpt)
    print(f"\nSaved final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()

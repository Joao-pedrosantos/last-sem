"""
Training script for the CXR pneumonia classifier — single-GPU or multi-GPU DDP.

Identical preprocessing to the serving backend (lung-segmented crop via
`cxr_pipeline.lung_focused_crop`) so train/serve drift is impossible by
construction.

Outputs `outputs/pneumonia_model_final.pt` with `img_size` and `threshold`
metadata embedded; backend/main.py reads those keys at boot.

Usage:
    # Single GPU
    python backend/train.py

    # Multi-GPU on one node (e.g. 4 V100s) — launches one process per GPU.
    torchrun --nproc_per_node=4 --standalone backend/train.py \
        --batch-size 32 --num-workers 8

Recommended: run `python backend/build_cache.py` once first to pre-populate
`outputs/preproc_cache/`. Multi-GPU training does NOT cope well with workers
lazily segmenting on different CUDA devices.

Perf knobs already on:
    - AMP fp16
    - channels-last memory format
    - torch.compile (disable with --no-compile)
    - DistributedSampler + NCCL all-reduce when launched via torchrun
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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


# --- distributed helpers --------------------------------------------------


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def log(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs)


def setup_dist() -> torch.device:
    """Init NCCL if launched under torchrun. Returns this rank's device."""
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_dist() -> None:
    if is_dist():
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist():
        dist.barrier()


# --- module-unwrap (DDP + torch.compile) ----------------------------------


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile wrappers to access the raw module."""
    m = model
    while hasattr(m, "module") and not isinstance(m, PneumoniaClassifier):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


# --- data -----------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RSNADataset(Dataset):
    """
    Lung-segmented RSNA pneumonia dataset.

    Reads cached crops from `outputs/preproc_cache/<pid>.npy`. If the cache
    is missing, falls back to computing it (slow + bad with DDP workers).
    Run `build_cache.py` beforehand.
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
    return df.groupby("patientId", as_index=False).agg(target=("Target", "max"))


# --- model freeze/unfreeze ------------------------------------------------


def freeze_backbone(model: nn.Module) -> None:
    raw = unwrap(model)
    for p in raw.model.parameters():
        p.requires_grad = False
    for p in raw.model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n: int = 2) -> None:
    """EfficientNet-B4 timm layout: model.blocks is the sequential MBConv stack."""
    raw = unwrap(model)
    blocks = raw.model.blocks
    for p in raw.model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks) - n), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    for p in raw.model.get_classifier().parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- train / eval ---------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    sampler, epoch_idx) -> float:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch_idx)
    total_loss = torch.zeros(1, device=device)
    n = torch.zeros(1, device=device)
    for imgs, targets in loader:
        imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(imgs).view(-1)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * imgs.size(0)
        n += imgs.size(0)
    if is_dist():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
    return (total_loss / n.clamp(min=1)).item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run on rank 0 only — caller is responsible for not invoking elsewhere."""
    model.eval()
    total_loss, n = 0.0, 0
    probs_all, labels_all = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(imgs).view(-1)
            loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
        labels_all.append(targets.cpu().numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    auc = roc_auc_score(labels, probs)
    return total_loss / max(n, 1), auc, probs, labels


def run_phase(name, model, train_loader, val_loader, criterion, optimizer,
              scheduler, device, scaler, epochs, best_auc, best_ckpt, sampler):
    log(f"\n=== {name} === ({count_trainable(model)/1e6:.2f}M trainable params)")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, criterion, optimizer,
                             device, scaler, sampler, epoch)
        scheduler.step()
        if is_main():
            vl, vauc, _, _ = evaluate(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]
            flag = ""
            if vauc > best_auc:
                best_auc = vauc
                torch.save(unwrap(model).state_dict(), best_ckpt)
                flag = " *"
            print(f"  Epoch {epoch:2d}/{epochs} | TL {tl:.4f} | VL {vl:.4f} | "
                  f"VAUC {vauc:.4f} | LR {lr:.2e} | {time.time()-t0:.0f}s{flag}")
        barrier()

    if is_dist():
        b = torch.tensor([best_auc], device=device, dtype=torch.float64)
        dist.broadcast(b, src=0)
        best_auc = b.item()
    return best_auc


# --- main -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-phase1", type=int, default=5)
    ap.add_argument("--epochs-phase2", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Per-GPU batch size.")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--lr-phase1", type=float, default=1e-4)
    ap.add_argument("--lr-phase2", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers per rank.")
    ap.add_argument("--no-compile", action="store_true",
                    help="Disable torch.compile (use if it hits a graph break bug).")
    ap.add_argument("--no-lr-scale", action="store_true",
                    help="Do NOT linearly scale LR by world_size in DDP.")
    args = ap.parse_args()

    device = setup_dist()
    set_seed(SEED + get_rank())

    if is_main():
        OUT_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        print(f"Device:     {device} | torch {torch.__version__}")
        print(f"World size: {get_world_size()}")
        if device.type == "cuda":
            print(f"GPU:        {torch.cuda.get_device_name(device.index or 0)}")
    barrier()

    # Linear-scale LR with effective batch size when in DDP.
    if is_dist() and not args.no_lr_scale:
        ws = get_world_size()
        args.lr_phase1 *= ws
        args.lr_phase2 *= ws
        log(f"LR linearly scaled by world_size={ws}: "
            f"phase1={args.lr_phase1:.2e}  phase2={args.lr_phase2:.2e}")

    df = build_dataframe()
    log(f"Total samples: {len(df):,} (positive rate {df['target'].mean():.3f})")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["target"], random_state=SEED
    )
    log(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_tf = make_train_transforms(args.img_size)
    val_tf = make_val_transforms(args.img_size)
    train_ds = RSNADataset(train_df, TRAIN_IMG_DIR, train_tf, args.img_size)
    val_ds = RSNADataset(val_df, TRAIN_IMG_DIR, val_tf, args.img_size)
    test_ds = RSNADataset(test_df, TRAIN_IMG_DIR, val_tf, args.img_size)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, sampler=train_sampler,
        shuffle=(train_sampler is None), **common,
    )
    # Val/test only used on rank 0 — plain sequential samplers.
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    raw_model = PneumoniaClassifier("efficientnet_b4", pretrained=True, dropout=0.3)
    raw_model = raw_model.to(device, memory_format=torch.channels_last)

    if is_dist():
        # find_unused_parameters=True because Phase 1 freezes the backbone:
        # those params still run in forward but don't receive grads.
        model = DDP(raw_model, device_ids=[device.index],
                    find_unused_parameters=True)
    else:
        model = raw_model

    if not args.no_compile:
        model = torch.compile(model)

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
        best_auc, best_ckpt, train_sampler,
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
        best_auc, best_ckpt, train_sampler,
    )

    # --- Final eval + save (rank 0 only)
    if is_main():
        unwrap(model).load_state_dict(torch.load(best_ckpt, map_location=device))
        _, val_auc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        _, test_auc, _, _ = evaluate(model, test_loader, criterion, device)
        threshold = threshold_at_specificity(val_labels, val_probs, target_spec=0.95)
        print(f"\nFinal Val AUC:  {val_auc:.4f}")
        print(f"Final Test AUC: {test_auc:.4f}")
        print(f"Clinical threshold (val Spec=0.95): {threshold:.4f}")

        final_ckpt = OUT_DIR / "pneumonia_model_final.pt"
        torch.save({
            "model_state_dict": unwrap(model).state_dict(),
            "img_size": args.img_size,
            "threshold": float(threshold),
            "model_name": "efficientnet_b4",
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        }, final_ckpt)
        print(f"\nSaved final checkpoint: {final_ckpt}")

    barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()

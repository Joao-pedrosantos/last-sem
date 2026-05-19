"""
Pre-populate `outputs/preproc_cache/<patientId>.npy` with lung-segmented crops.

Run this once before training — especially before multi-GPU training, where
having DataLoader workers spawn the PSPNet on each rank's GPU is wasteful and
fragile (CUDA-after-fork). With the cache pre-built, training workers do pure
CPU work (np.load + augmentation) and parallelize cleanly across ranks.

Idempotent: skips PIDs that already have a .npy in the cache.

Usage:
    python backend/build_cache.py                     # all PIDs
    python backend/build_cache.py --limit 100         # first 100 (smoke test)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cxr_pipeline import lung_focused_crop, read_dicom  # noqa: E402

DATA_DIR = ROOT / "rsna-pneumonia-detection-challenge"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "preproc_cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N PIDs (smoke test).")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(LABELS_CSV)
    pids = df["patientId"].unique().tolist()
    if args.limit is not None:
        pids = pids[: args.limit]
    print(f"PIDs to consider: {len(pids):,}")

    done = skipped = failed = 0
    t0 = time.time()
    last_log = t0
    for i, pid in enumerate(pids, start=1):
        cache_path = CACHE_DIR / f"{pid}.npy"
        if cache_path.exists():
            skipped += 1
        else:
            try:
                img = read_dicom(TRAIN_IMG_DIR / f"{pid}.dcm")
                cropped, _ = lung_focused_crop(img)
                np.save(cache_path, cropped)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  failed {pid}: {e}")

        now = time.time()
        if now - last_log >= 10.0 or i == len(pids):
            rate = i / max(now - t0, 1e-6)
            eta = (len(pids) - i) / max(rate, 1e-6)
            print(f"  [{i:>5}/{len(pids)}]  done={done} skip={skipped} fail={failed}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}m")
            last_log = now

    print(f"\nFinished in {(time.time()-t0)/60:.1f} min — "
          f"done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()

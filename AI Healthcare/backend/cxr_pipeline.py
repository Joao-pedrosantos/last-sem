"""
Unified CXR preprocessing — imported by both training (project.ipynb) and
serving (backend/main.py). Train/serve drift is the bug we are fixing; this
module is the only place these transforms are defined.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import numpy as np
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pydicom.pixels import apply_voi_lut

ImageNetMean = (0.485, 0.456, 0.406)
ImageNetStd = (0.229, 0.224, 0.225)


def read_dicom(source: Union[bytes, bytearray, str, Path]) -> np.ndarray:
    """
    Read a DICOM into a uint8 H×W grayscale array with lung tissue bright.

    Precedence for pixel mapping:
        1. VOI LUT (via pydicom.pixels.apply_voi_lut) if present.
        2. Otherwise the raw pixel values, percentile-clipped at 1/99 to drop
           burn-in extremes.
    Then we rescale to [0, 255]. Finally, if PhotometricInterpretation is
    MONOCHROME1, invert so high values = bright tissue (lung-positive polarity).
    """
    if isinstance(source, (bytes, bytearray)):
        ds = pydicom.dcmread(io.BytesIO(bytes(source)))
    else:
        ds = pydicom.dcmread(str(source))

    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)

    lo, hi = np.percentile(arr, [1.0, 99.0])
    arr = np.clip(arr, lo, hi)
    span = max(float(hi - lo), 1e-6)
    arr = (arr - lo) / span * 255.0

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = 255.0 - arr

    return arr.astype(np.uint8)


def lung_roi_crop(img: np.ndarray) -> np.ndarray:
    """
    Crop to the lung field via Otsu thresholding on a downsampled copy.
    Falls back to a fixed 8% inset if the detected blob is too small or has
    a degenerate aspect ratio. Output is padded to a square.

    Deterministic: same input → same output. Never raises.
    """
    assert img.ndim == 2 and img.dtype == np.uint8, "expect uint8 H×W grayscale"
    h, w = img.shape

    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels < 2:
        return _fallback_inset(img)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    x0s = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y0s = int(stats[largest_idx, cv2.CC_STAT_TOP])
    bws = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    bhs = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])

    area_frac = (bws * bhs) / (256.0 * 256.0)
    aspect = bws / max(bhs, 1)
    if area_frac < 0.35 or area_frac > 0.95 or not (0.5 <= aspect <= 2.0):
        return _fallback_inset(img)

    sy, sx = h / 256.0, w / 256.0
    y0 = int(round(y0s * sy))
    x0 = int(round(x0s * sx))
    y1 = min(h, int(round((y0s + bhs) * sy)))
    x1 = min(w, int(round((x0s + bws) * sx)))
    crop = img[y0:y1, x0:x1]
    return _pad_to_square(crop)


def _fallback_inset(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    dy = int(round(h * 0.08))
    dx = int(round(w * 0.08))
    return _pad_to_square(img[dy:h - dy, dx:w - dx])


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    py_top = (side - h) // 2
    py_bot = side - h - py_top
    px_lt = (side - w) // 2
    px_rt = side - w - px_lt
    return np.pad(img, ((py_top, py_bot), (px_lt, px_rt)), mode="constant", constant_values=0)


def to_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Resize grayscale to (size, size) and replicate to 3 channels."""
    assert img.ndim == 2, "expect H×W grayscale"
    pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    arr = np.array(pil)
    return np.stack([arr, arr, arr], axis=-1)


def make_val_transforms(size: int) -> A.Compose:
    """Inference / validation transforms — normalize only, no augmentation."""
    return A.Compose([
        A.Normalize(mean=ImageNetMean, std=ImageNetStd),
        ToTensorV2(),
    ])


def make_train_transforms(size: int) -> A.Compose:
    """Training augmentations chosen to suppress border shortcuts (CoarseDropout)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 64),
                        hole_width_range=(16, 64), fill=0, p=0.3),
        A.Normalize(mean=ImageNetMean, std=ImageNetStd),
        ToTensorV2(),
    ])


def tta_predict(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    """
    Test-time augmentation: average sigmoid(model(x)) and sigmoid(model(flip_h(x))).
    Expects a single-image batch tensor (1, 3, H, W).
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1, "expect a 1-image batch"
    model.eval()
    with torch.no_grad():
        p1 = torch.sigmoid(model(tensor)).item()
        p2 = torch.sigmoid(model(torch.flip(tensor, dims=[-1]))).item()
    return (p1 + p2) / 2.0

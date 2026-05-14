# CXR Pipeline Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the RSNA pneumonia model's image pipeline so training and inference share one preprocessing source of truth, eliminating the border-attention shortcut and producing clinically-aligned Grad-CAM heatmaps.

**Architecture:** Three new backend modules (`cxr_pipeline.py`, `training.py`, `grad_cam_utils.py`) imported by both the training notebook and the FastAPI serving layer. Notebook cells updated to use them; `main.py` stripped of post-hoc border hacks. Retrain on the cohort-filtered dataset with focal loss, cosine LR, and Grad-CAM++.

**Tech Stack:** PyTorch + timm (EfficientNet-B4), pydicom, albumentations, opencv-python, scipy, pytorch-grad-cam, FastAPI, pytest.

---

## File Structure

**Create:**
- `AI Healthcare/backend/cxr_pipeline.py` — `read_dicom`, `lung_roi_crop`, `to_rgb`, `make_train_transforms`, `make_val_transforms`, `tta_predict`
- `AI Healthcare/backend/training.py` — `FocalLoss`, `threshold_at_specificity`
- `AI Healthcare/backend/grad_cam_utils.py` — `gradcam_pp_heatmap`, `bbox_from_heatmap_percentile`, `target_layer_for_efficientnet_b4`
- `AI Healthcare/backend/tests/__init__.py` (empty)
- `AI Healthcare/backend/tests/conftest.py` (puts `backend/` on `sys.path`)
- `AI Healthcare/backend/tests/test_cxr_pipeline.py`
- `AI Healthcare/backend/tests/test_training.py`
- `AI Healthcare/backend/tests/test_grad_cam_utils.py`

**Modify:**
- `AI Healthcare/backend/main.py` — drop border hacks, use new pipeline + grad_cam_utils, read threshold from checkpoint
- `AI Healthcare/backend/preprocessing.py` — thin compat shim re-exporting from `cxr_pipeline`
- `AI Healthcare/backend/model.py` — return `img_size`/`threshold` from `load_model`
- `AI Healthcare/backend/requirements.txt` — add `pytest`, `opencv-python-headless`, `scipy`, `scikit-learn`
- `AI Healthcare/project.ipynb` — replace dataset/transforms/loss/scheduler/Grad-CAM cells; add cohort-filter cell; persist threshold to checkpoint

**Run pytest from:** `AI Healthcare/` (so `backend/` is importable as `backend.*`).

---

## Task 1: Set up test infrastructure

**Files:**
- Create: `AI Healthcare/backend/tests/__init__.py`
- Create: `AI Healthcare/backend/tests/conftest.py`
- Modify: `AI Healthcare/backend/requirements.txt`

- [ ] **Step 1: Create empty `__init__.py`**

Create `AI Healthcare/backend/tests/__init__.py` with empty content.

- [ ] **Step 2: Create `conftest.py` to expose backend on sys.path**

Create `AI Healthcare/backend/tests/conftest.py`:

```python
import sys
from pathlib import Path

# Put the project root (parent of backend/) on sys.path so tests can do
# `from backend.cxr_pipeline import ...` regardless of where pytest is invoked.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Add test + image-processing deps to requirements**

Update `AI Healthcare/backend/requirements.txt` to:

```
fastapi
uvicorn[standard]
python-multipart
torch
torchvision
pydicom
albumentations
Pillow
grad-cam
numpy
timm
opencv-python-headless
scipy
scikit-learn
pytest
```

- [ ] **Step 4: Install deps**

Run from `AI Healthcare/`:

```bash
source venv/bin/activate && pip install -r backend/requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 5: Verify pytest discovers the tests dir**

Run from `AI Healthcare/`:

```bash
pytest backend/tests --collect-only -q
```

Expected: `no tests collected` (exit 5) — no tests yet, but pytest must run without import errors.

- [ ] **Step 6: Commit**

```bash
git add "AI Healthcare/backend/tests/__init__.py" \
        "AI Healthcare/backend/tests/conftest.py" \
        "AI Healthcare/backend/requirements.txt"
git commit -m "test: scaffold backend tests directory and deps"
```

---

## Task 2: Implement `read_dicom` in `cxr_pipeline.py`

**Files:**
- Create: `AI Healthcare/backend/cxr_pipeline.py`
- Create: `AI Healthcare/backend/tests/test_cxr_pipeline.py`

- [ ] **Step 1: Write the failing test for MONOCHROME1 inversion**

Create `AI Healthcare/backend/tests/test_cxr_pipeline.py`:

```python
import io
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian


def _make_dicom(pixels: np.ndarray, photometric: str = "MONOCHROME2",
                window_center: float | None = None, window_width: float | None = None) -> bytes:
    """Build a minimal in-memory DICOM with the given pixel array."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    if window_center is not None:
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_read_dicom_inverts_monochrome1():
    from backend.cxr_pipeline import read_dicom
    # Gradient pixels: low at top, high at bottom
    pixels = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME1")
    img = read_dicom(raw)
    assert img.dtype == np.uint8
    assert img.shape == (16, 16)
    # MONOCHROME1: high pixel values represent dark — after inversion, top row
    # (originally low) should be bright, bottom row should be dark
    assert img[0, 0] > img[-1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py::test_read_dicom_inverts_monochrome1 -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.cxr_pipeline'`.

- [ ] **Step 3: Add MONOCHROME2 and percentile-clip tests**

Append to `AI Healthcare/backend/tests/test_cxr_pipeline.py`:

```python
def test_read_dicom_monochrome2_is_not_inverted():
    from backend.cxr_pipeline import read_dicom
    pixels = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME2")
    img = read_dicom(raw)
    # Top row (low values) stays dark, bottom row (high) stays bright
    assert img[0, 0] < img[-1, 0]


def test_read_dicom_returns_uint8_in_full_range():
    from backend.cxr_pipeline import read_dicom
    # Uniform gradient — output should span close to [0, 255]
    pixels = np.tile(np.linspace(0, 65535, 32, dtype=np.uint16), (32, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME2")
    img = read_dicom(raw)
    assert img.min() <= 5
    assert img.max() >= 250
```

- [ ] **Step 4: Run all three tests to verify they fail**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v
```

Expected: 3 FAIL (import errors).

- [ ] **Step 5: Implement `read_dicom`**

Create `AI Healthcare/backend/cxr_pipeline.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v
```

Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add "AI Healthcare/backend/cxr_pipeline.py" \
        "AI Healthcare/backend/tests/test_cxr_pipeline.py"
git commit -m "feat(cxr): add robust DICOM reader with VOI LUT + MONOCHROME1 handling"
```

---

## Task 3: Implement `lung_roi_crop`

**Files:**
- Modify: `AI Healthcare/backend/cxr_pipeline.py`
- Modify: `AI Healthcare/backend/tests/test_cxr_pipeline.py`

- [ ] **Step 1: Write failing tests for lung_roi_crop**

Append to `AI Healthcare/backend/tests/test_cxr_pipeline.py`:

```python
def test_lung_roi_crop_strips_black_border():
    from backend.cxr_pipeline import lung_roi_crop
    # 1024x1024 image with a 100-px black border and a uniform-bright center
    img = np.zeros((1024, 1024), dtype=np.uint8)
    img[100:924, 100:924] = 200
    cropped = lung_roi_crop(img)
    # Output should be square, close to the inner 824x824 region (±20 px slack)
    assert cropped.shape[0] == cropped.shape[1]  # square
    assert 800 <= cropped.shape[0] <= 850


def test_lung_roi_crop_falls_back_on_uniform_image():
    from backend.cxr_pipeline import lung_roi_crop
    # Uniform image — Otsu produces a degenerate mask; fallback to 8% inset
    img = np.full((1024, 1024), 128, dtype=np.uint8)
    cropped = lung_roi_crop(img)
    # 8% inset on a 1024 image leaves 1024 - 2*81 ≈ 862 px; padded to square
    assert cropped.shape[0] == cropped.shape[1]
    assert 850 <= cropped.shape[0] <= 870


def test_lung_roi_crop_falls_back_on_thin_blob():
    from backend.cxr_pipeline import lung_roi_crop
    # Thin vertical bar — aspect ratio < 0.5, should trigger fallback
    img = np.zeros((1024, 1024), dtype=np.uint8)
    img[200:824, 500:520] = 255  # 624 tall, 20 wide
    cropped = lung_roi_crop(img)
    # Fallback gives ~862 square, not the 20-wide bar
    assert cropped.shape[1] > 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v -k lung_roi_crop
```

Expected: 3 FAIL with `ImportError: cannot import name 'lung_roi_crop'`.

- [ ] **Step 3: Implement `lung_roi_crop` + private helpers**

Append to `AI Healthcare/backend/cxr_pipeline.py`:

```python
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
    if area_frac < 0.35 or not (0.5 <= aspect <= 2.0):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/cxr_pipeline.py" \
        "AI Healthcare/backend/tests/test_cxr_pipeline.py"
git commit -m "feat(cxr): add Otsu-based lung_roi_crop with fixed-inset fallback"
```

---

## Task 4: Implement `to_rgb` and transforms

**Files:**
- Modify: `AI Healthcare/backend/cxr_pipeline.py`
- Modify: `AI Healthcare/backend/tests/test_cxr_pipeline.py`

- [ ] **Step 1: Write failing tests**

Append to `AI Healthcare/backend/tests/test_cxr_pipeline.py`:

```python
def test_to_rgb_shape_and_dtype():
    from backend.cxr_pipeline import to_rgb
    img = np.full((824, 824), 128, dtype=np.uint8)
    out = to_rgb(img, size=640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8
    # All 3 channels identical (grayscale replication)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_make_val_transforms_produces_tensor():
    import torch
    from backend.cxr_pipeline import make_val_transforms
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    out = make_val_transforms(640)(image=img)["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 640, 640)
    assert out.dtype == torch.float32


def test_make_train_transforms_is_stochastic():
    import torch
    from backend.cxr_pipeline import make_train_transforms
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    tf = make_train_transforms(640)
    out1 = tf(image=img)["image"]
    # Run many times — at least one should differ (CoarseDropout / flip etc.)
    differs = any(not torch.equal(tf(image=img)["image"], out1) for _ in range(20))
    assert differs
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v -k "to_rgb or transforms"
```

Expected: 3 FAIL (missing imports).

- [ ] **Step 3: Implement `to_rgb`, `make_val_transforms`, `make_train_transforms`**

Append to `AI Healthcare/backend/cxr_pipeline.py`:

```python
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
        A.CoarseDropout(max_holes=4, max_height=64, max_width=64,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=0, p=0.3),
        A.Normalize(mean=ImageNetMean, std=ImageNetStd),
        ToTensorV2(),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/cxr_pipeline.py" \
        "AI Healthcare/backend/tests/test_cxr_pipeline.py"
git commit -m "feat(cxr): add to_rgb + train/val transform builders"
```

---

## Task 5: Implement `tta_predict`

**Files:**
- Modify: `AI Healthcare/backend/cxr_pipeline.py`
- Modify: `AI Healthcare/backend/tests/test_cxr_pipeline.py`

- [ ] **Step 1: Write failing test**

Append to `AI Healthcare/backend/tests/test_cxr_pipeline.py`:

```python
def test_tta_predict_averages_original_and_flip():
    import torch
    import torch.nn as nn
    from backend.cxr_pipeline import tta_predict

    class FlipSensitive(nn.Module):
        """Returns sum of left half — original vs hflip give different sums."""
        def forward(self, x):
            left = x[:, :, :, :x.shape[-1] // 2].sum()
            return left.view(1, 1)

    model = FlipSensitive().eval()
    # Asymmetric input — left half is bright, right half is zero
    t = torch.zeros(1, 3, 8, 8)
    t[:, :, :, :4] = 1.0
    p = tta_predict(model, t)
    # Original logit = 96 (sum of left = 3*8*4*1), sigmoid ≈ 1.0
    # Flipped logit = 0, sigmoid = 0.5
    # Mean ≈ 0.75
    assert 0.7 < p < 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v -k tta_predict
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `tta_predict`**

Append to `AI Healthcare/backend/cxr_pipeline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_cxr_pipeline.py -v
```

Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/cxr_pipeline.py" \
        "AI Healthcare/backend/tests/test_cxr_pipeline.py"
git commit -m "feat(cxr): add tta_predict (original + horizontal flip averaging)"
```

---

## Task 6: Implement `FocalLoss` in `training.py`

**Files:**
- Create: `AI Healthcare/backend/training.py`
- Create: `AI Healthcare/backend/tests/test_training.py`

- [ ] **Step 1: Write failing tests**

Create `AI Healthcare/backend/tests/test_training.py`:

```python
import numpy as np
import torch
import torch.nn.functional as F


def test_focal_loss_equals_bce_when_gamma_zero_and_alpha_half():
    from backend.training import FocalLoss
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    focal = FocalLoss(alpha=0.5, gamma=0.0, label_smoothing=0.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean") * 0.5
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    from backend.training import FocalLoss
    # Two easy examples (confident, correct) and one hard (wrong)
    logits_easy = torch.tensor([5.0, -5.0])
    targets_easy = torch.tensor([1.0, 0.0])
    logits_hard = torch.tensor([-5.0, 5.0])
    targets_hard = torch.tensor([1.0, 0.0])  # both wrong

    focal = FocalLoss(alpha=0.5, gamma=2.0)
    loss_easy = focal(logits_easy, targets_easy)
    loss_hard = focal(logits_hard, targets_hard)
    assert loss_hard > 100 * loss_easy  # focal massively up-weights hard examples
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_training.py -v
```

Expected: 2 FAIL with ImportError.

- [ ] **Step 3: Implement `FocalLoss`**

Create `AI Healthcare/backend/training.py`:

```python
"""
Training-time utilities: focal loss and clinical-threshold helper.
Used by the notebook, not the serving backend.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced single-logit classifiers.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(logit) for target=1, else 1 - sigmoid(logit).

    Label smoothing pulls targets toward 0.5 by `label_smoothing` units before
    computing pt — softens overconfidence without changing class balance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.label_smoothing > 0:
            ls = self.label_smoothing
            targets = targets * (1.0 - ls) + 0.5 * ls

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_training.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/training.py" \
        "AI Healthcare/backend/tests/test_training.py"
git commit -m "feat(training): add binary FocalLoss with label smoothing"
```

---

## Task 7: Implement `threshold_at_specificity`

**Files:**
- Modify: `AI Healthcare/backend/training.py`
- Modify: `AI Healthcare/backend/tests/test_training.py`

- [ ] **Step 1: Write failing test**

Append to `AI Healthcare/backend/tests/test_training.py`:

```python
def test_threshold_at_specificity_picks_correct_cutoff():
    from backend.training import threshold_at_specificity
    # Synthetic: 100 negatives with probs uniform in [0, 0.5],
    # 100 positives with probs uniform in [0.5, 1.0].
    rng = np.random.default_rng(0)
    y_neg = rng.uniform(0.0, 0.5, 100)
    y_pos = rng.uniform(0.5, 1.0, 100)
    y_proba = np.concatenate([y_neg, y_pos])
    y_true = np.concatenate([np.zeros(100), np.ones(100)])

    thr = threshold_at_specificity(y_true, y_proba, target_spec=0.95)
    # 95% spec means cutoff that excludes 95% of negatives — near 0.475
    assert 0.4 < thr < 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_training.py -v -k threshold
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `threshold_at_specificity`**

Append to `AI Healthcare/backend/training.py`:

```python
def threshold_at_specificity(y_true: np.ndarray, y_proba: np.ndarray,
                             target_spec: float = 0.95) -> float:
    """
    Return the probability threshold whose specificity is closest to `target_spec`
    on the ROC curve. Used to pick the clinical operating point.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificities = 1.0 - fpr
    idx = int(np.argmin(np.abs(specificities - target_spec)))
    return float(thresholds[idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_training.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/training.py" \
        "AI Healthcare/backend/tests/test_training.py"
git commit -m "feat(training): add threshold_at_specificity helper"
```

---

## Task 8: Implement Grad-CAM helpers

**Files:**
- Create: `AI Healthcare/backend/grad_cam_utils.py`
- Create: `AI Healthcare/backend/tests/test_grad_cam_utils.py`

- [ ] **Step 1: Write failing tests**

Create `AI Healthcare/backend/tests/test_grad_cam_utils.py`:

```python
import numpy as np


def test_bbox_percentile_picks_top_activations():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    # 512×512 heatmap, low everywhere except a 50×50 spot at (100, 200)
    heatmap = np.full((512, 512), 0.1, dtype=np.float32)
    heatmap[100:150, 200:250] = 0.9
    bbox = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=0.0, size=512)
    assert bbox is not None
    # Centroid should be inside the hot spot (allow ±15 px for percentile slack)
    cx = bbox["x"] + bbox["width"] // 2
    cy = bbox["y"] + bbox["height"] // 2
    assert 200 - 15 <= cx <= 250 + 15
    assert 100 - 15 <= cy <= 150 + 15


def test_bbox_percentile_returns_none_for_flat_heatmap():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    heatmap = np.zeros((512, 512), dtype=np.float32)
    assert bbox_from_heatmap_percentile(heatmap, percentile=90.0) is None


def test_bbox_percentile_blur_widens_bbox():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    heatmap = np.full((512, 512), 0.1, dtype=np.float32)
    heatmap[250:260, 250:260] = 1.0
    bbox_sharp = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=0.0)
    bbox_blurred = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=8.0)
    assert bbox_blurred["width"] > bbox_sharp["width"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_grad_cam_utils.py -v
```

Expected: 3 FAIL with ImportError.

- [ ] **Step 3: Implement Grad-CAM helpers**

Create `AI Healthcare/backend/grad_cam_utils.py`:

```python
"""
Grad-CAM helpers shared by the training notebook and the serving backend.

Choices:
- Algorithm: GradCAM++ — gives smoother, less saturating heatmaps than GradCAM.
- Target layer: last MBConv block of EfficientNet-B4 — higher feature resolution
  than the post-conv_head bn2.
- Bbox extraction: threshold at the 90th percentile of heatmap values (not 50%
  of max) so a tiny saturated peak doesn't define a huge region.
- Pre-bbox Gaussian blur prevents noisy speckle from creating jagged regions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def bbox_from_heatmap_percentile(
    heatmap: np.ndarray,
    percentile: float = 90.0,
    blur_sigma: float = 8.0,
    size: int = 512,
) -> Optional[dict]:
    """
    Convert a (H, W) heatmap in [0, 1] to a bounding box dict in `size`-space.
    Returns None if the heatmap has no signal (all zeros).
    """
    if heatmap.size == 0:
        return None

    if heatmap.shape != (size, size):
        pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((size, size), Image.BILINEAR)
        heatmap = np.array(pil).astype(np.float32) / 255.0

    if blur_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

    threshold = float(np.percentile(heatmap, percentile))
    if threshold <= 0:
        return None

    mask = heatmap >= threshold
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}


def target_layer_for_efficientnet_b4(model_wrapper):
    """
    Return the last MBConv stage of an EfficientNet-B4 timm model wrapped in
    a `PneumoniaClassifier`-style class (`.model` attribute holds the timm net).
    """
    return model_wrapper.model.blocks[-1]


def gradcam_pp_heatmap(model_wrapper, tensor, target_layer) -> np.ndarray:
    """
    Run GradCAM++ on a single-image batch tensor. Returns a (H, W) heatmap in [0, 1].
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    cam = GradCAMPlusPlus(model=model_wrapper, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)
    return grayscale_cam[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_grad_cam_utils.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/backend/grad_cam_utils.py" \
        "AI Healthcare/backend/tests/test_grad_cam_utils.py"
git commit -m "feat(grad-cam): add GradCAM++ wrapper + percentile bbox + blur"
```

---

## Task 9: Refactor `backend/main.py` to use the new pipeline

**Files:**
- Modify: `AI Healthcare/backend/main.py`
- Modify: `AI Healthcare/backend/preprocessing.py`

- [ ] **Step 1: Replace `backend/main.py` with the new pipeline-driven version**

Overwrite `AI Healthcare/backend/main.py` with:

```python
import base64
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from cxr_pipeline import (
    lung_roi_crop,
    make_val_transforms,
    read_dicom,
    to_rgb,
    tta_predict,
)
from grad_cam_utils import (
    bbox_from_heatmap_percentile,
    gradcam_pp_heatmap,
    target_layer_for_efficientnet_b4,
)
from model import load_model
from preprocessing import load_image as _load_standard_image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULT_MODEL_CANDIDATES = [
    "models/best_model.pt",
    "../outputs/best_model.pt",
]


def _resolve_default_model_path() -> str:
    for candidate in _DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return _DEFAULT_MODEL_CANDIDATES[0]


MODEL_PATH = os.getenv("MODEL_PATH", _resolve_default_model_path())
# The model checkpoint embeds its operating threshold. We only fall back to the
# env var (or 0.5) when the checkpoint doesn't carry one.
FALLBACK_THRESHOLD = float(os.getenv("CLINICAL_THRESHOLD", "0.5"))

DISPLAY_SIZE = 512  # frontend canvas
DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}

# ---------------------------------------------------------------------------
# App & model startup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray pneumonia classifier (EfficientNet-B4) — DICOM + standard image formats.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model, weights_loaded, model_meta = load_model(MODEL_PATH, DEVICE)
INFERENCE_SIZE = int(model_meta.get("img_size", 640))
CLINICAL_THRESHOLD = float(model_meta.get("threshold", FALLBACK_THRESHOLD))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dicom(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in DICOM_EXTENSIONS


def _array_to_base64_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _display_image(img_gray: np.ndarray) -> np.ndarray:
    """Resize the lung-cropped grayscale image to DISPLAY_SIZE×DISPLAY_SIZE RGB for the frontend."""
    pil = Image.fromarray(img_gray).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    arr = np.array(pil)
    return np.stack([arr, arr, arr], axis=-1)


def _mock_response() -> dict:
    """Placeholder response when weights aren't loaded — keeps the frontend wired."""
    h = w = DISPLAY_SIZE
    canvas = np.full((h, w, 3), 80, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, sigma = int(w * 0.62), int(h * 0.55), 95
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    r = np.clip(canvas[:, :, 0].astype(np.int32) + (blob * 200).astype(np.int32), 0, 255).astype(np.uint8)
    g = np.clip(canvas[:, :, 1].astype(np.int32) + (blob * 80).astype(np.int32), 0, 255).astype(np.uint8)
    b = canvas[:, :, 2]
    heatmap_rgb = np.stack([r, g, b], axis=-1)
    return {
        "prediction": "pneumonia",
        "probability": 0.78,
        "threshold": CLINICAL_THRESHOLD,
        "weights_loaded": False,
        "mock": True,
        "image": {
            "base64": _array_to_base64_png(heatmap_rgb),
            "width": DISPLAY_SIZE,
            "height": DISPLAY_SIZE,
        },
        "gradcam": _array_to_base64_png(heatmap_rgb),
        "bbox": bbox_from_heatmap_percentile(blob, percentile=90.0, blur_sigma=8.0, size=DISPLAY_SIZE),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "weights_loaded": weights_loaded,
        "model_path": MODEL_PATH,
        "threshold": CLINICAL_THRESHOLD,
        "inference_size": INFERENCE_SIZE,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True, description="Include Grad-CAM heatmap in response"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    if not weights_loaded:
        return _mock_response()

    # --- read + crop ---
    try:
        if _is_dicom(file.filename):
            img_gray = read_dicom(file_bytes)
        else:
            img_rgb = _load_standard_image(file_bytes)
            img_gray = np.array(Image.fromarray(img_rgb).convert("L"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc

    img_cropped = lung_roi_crop(img_gray)
    display_rgb = _display_image(img_cropped)

    # --- model input ---
    img_rgb_model = to_rgb(img_cropped, size=INFERENCE_SIZE)
    tensor = make_val_transforms(INFERENCE_SIZE)(image=img_rgb_model)["image"].unsqueeze(0).to(DEVICE)

    probability = tta_predict(model, tensor)
    prediction = "pneumonia" if probability >= CLINICAL_THRESHOLD else "normal"

    # --- Grad-CAM ---
    gradcam_b64 = None
    bbox = None
    try:
        heatmap = gradcam_pp_heatmap(
            model_wrapper=model,
            tensor=tensor,
            target_layer=target_layer_for_efficientnet_b4(model),
        )
        # Overlay onto the display image
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import Image as PILImage
        # Resize heatmap to display size for overlay
        heatmap_disp = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8))
            .resize((DISPLAY_SIZE, DISPLAY_SIZE), PILImage.BILINEAR)
        ).astype(np.float32) / 255.0
        img_float = display_rgb.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, heatmap_disp, use_rgb=True)
        gradcam_b64 = _array_to_base64_png(cam_image)
        bbox = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=8.0, size=DISPLAY_SIZE)
    except Exception:
        # Heatmap failure is non-fatal — prediction still returned.
        pass

    response = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": CLINICAL_THRESHOLD,
        "weights_loaded": True,
        "mock": False,
        "image": {
            "base64": _array_to_base64_png(display_rgb),
            "width": DISPLAY_SIZE,
            "height": DISPLAY_SIZE,
        },
        "bbox": bbox if prediction == "pneumonia" else None,
    }
    if include_gradcam:
        response["gradcam"] = gradcam_b64
    return response


# ---------------------------------------------------------------------------
# Frontend (kept for the legacy static index until the new frontend is merged)
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
```

- [ ] **Step 2: Replace `backend/preprocessing.py` with a thin compat layer**

Overwrite `AI Healthcare/backend/preprocessing.py` with:

```python
"""
Compatibility shim. The real preprocessing lives in cxr_pipeline.py — this
module is kept only to preserve the `load_image` entry point used by the
serving backend for standard (non-DICOM) image uploads.
"""
import io

import numpy as np
from PIL import Image


def load_image(file_bytes: bytes) -> np.ndarray:
    """Read a PNG/JPEG into a uint8 RGB numpy array."""
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
```

- [ ] **Step 3: Verify backend still imports cleanly**

Run from `AI Healthcare/backend/`:

```bash
python -c "import main; print('ok')"
```

Expected: `ok`. (May print model-loading warnings — that's fine.)

- [ ] **Step 4: Commit**

```bash
git add "AI Healthcare/backend/main.py" "AI Healthcare/backend/preprocessing.py"
git commit -m "refactor(backend): drive /predict through cxr_pipeline + grad_cam_utils"
```

---

## Task 10: Update `model.py` to surface checkpoint metadata

**Files:**
- Modify: `AI Healthcare/backend/model.py`

- [ ] **Step 1: Update `load_model` signature**

Overwrite `AI Healthcare/backend/model.py` with:

```python
import os
from typing import Tuple, Dict, Any

import timm
import torch
import torch.nn as nn


class PneumoniaClassifier(nn.Module):
    """
    Mirror of the training wrapper in project.ipynb so the saved state_dict
    (keys prefixed with `model.`) loads cleanly.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=1, drop_rate=dropout,
        )

    def forward(self, x):
        return self.model(x)


def load_model(weights_path: str, device: torch.device) -> Tuple[nn.Module, bool, Dict[str, Any]]:
    """
    Build the classifier and load saved weights if available.

    Returns (model, weights_loaded, meta) where meta is a dict of any extra
    fields embedded in the checkpoint — currently `img_size` and `threshold`.
    Meta is always a dict (possibly empty); the caller picks defaults for
    missing keys.
    """
    model = PneumoniaClassifier("efficientnet_b4", pretrained=False)

    weights_loaded = False
    meta: Dict[str, Any] = {}
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            for key in ("img_size", "threshold", "model_name"):
                if key in checkpoint:
                    meta[key] = checkpoint[key]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        weights_loaded = True

    model.eval().to(device)
    return model, weights_loaded, meta
```

- [ ] **Step 2: Verify imports**

Run from `AI Healthcare/backend/`:

```bash
python -c "from model import load_model; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add "AI Healthcare/backend/model.py"
git commit -m "refactor(model): surface img_size + threshold from checkpoint metadata"
```

---

## Task 11: Notebook — replace Dataset and transforms with cxr_pipeline

**Files:**
- Modify: `AI Healthcare/project.ipynb`

These edits use the notebook UI or `nbformat`. For each cell, locate it by its current first line and replace its source.

- [ ] **Step 1: Add `backend/` to notebook sys.path (new cell after cell 4 — Imports)**

Open the notebook. After the imports cell (cell 4), insert a new code cell:

```python
import sys
from pathlib import Path
backend_dir = Path.cwd() / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
```

- [ ] **Step 2: Replace cell 17 (transforms) entirely**

Find the cell starting `# 2.3  Image transforms (Albumentations)`. Replace its source with:

```python
# ============================================================
# 2.3  Image transforms — delegated to backend/cxr_pipeline.py
# ============================================================
from cxr_pipeline import make_train_transforms, make_val_transforms
IMG_SIZE = 640
train_transforms = make_train_transforms(IMG_SIZE)
val_transforms = make_val_transforms(IMG_SIZE)
```

- [ ] **Step 3: Replace cell 18 (Dataset) entirely**

Find the cell starting `# 2.4  PyTorch Dataset`. Replace its source with:

```python
# ============================================================
# 2.4  PyTorch Dataset — uses unified cxr_pipeline preprocessing
# ============================================================
from cxr_pipeline import read_dicom, lung_roi_crop, to_rgb

class RSNAPneumoniaDataset(Dataset):
    """RSNA pneumonia dataset. Loads DICOM → grayscale → lung-ROI crop → RGB → transforms."""

    def __init__(self, dataframe, img_dir, transforms, size=IMG_SIZE):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.img_dir / f"{row['patientId']}.dcm"
        gray = read_dicom(path)
        gray = lung_roi_crop(gray)
        rgb = to_rgb(gray, size=self.size)
        tensor = self.transforms(image=rgb)["image"]
        label = torch.tensor(row["target"], dtype=torch.float32)
        return tensor, label

train_ds = RSNAPneumoniaDataset(train_df, TRAIN_IMG_DIR, train_transforms)
val_ds   = RSNAPneumoniaDataset(val_df,   TRAIN_IMG_DIR, val_transforms)
test_ds  = RSNAPneumoniaDataset(test_df,  TRAIN_IMG_DIR, val_transforms)

print(f"Train dataset: {len(train_ds)} images")
print(f"Val dataset:   {len(val_ds)} images")
print(f"Test dataset:  {len(test_ds)} images")
```

- [ ] **Step 4: Insert cohort-filter cell before cell 15 (split)**

Find the cell starting `# 2.1  Stratified train/val/test split`. Insert a new cell directly *before* it:

```python
# ============================================================
# 2.0  Cohort filter — drop "No Lung Opacity / Not Normal"
# ============================================================
# These patients have other pathologies (cardiomegaly, effusion, atelectasis)
# that look like pneumonia and push the model toward non-pathological cues.
# We keep them aside as a separate stress-eval bucket on the test set.
stress_df  = df[df["class"] == "No Lung Opacity / Not Normal"].reset_index(drop=True)
df         = df[df["class"] != "No Lung Opacity / Not Normal"].reset_index(drop=True)
print(f"Stress (excluded from train/val): {len(stress_df)}")
print(f"Train/val/test pool:              {len(df)}")
```

- [ ] **Step 5: Run cells 0–19 end-to-end and verify a batch loads**

Restart the kernel. Run from the top through the DataLoaders cell (cell 19). Expected output of cell 19:

```
Batch images shape:  torch.Size([16, 3, 640, 640])
Batch labels shape:  torch.Size([16])
```

- [ ] **Step 6: Commit**

```bash
git add "AI Healthcare/project.ipynb"
git commit -m "notebook: route dataset + transforms through cxr_pipeline + drop Not-Normal cohort"
```

---

## Task 12: Notebook — switch to focal loss + cosine LR

**Files:**
- Modify: `AI Healthcare/project.ipynb`

- [ ] **Step 1: Replace cell 16 (pos_weight) entirely**

Find the cell starting `# 2.2  Compute class weights for Weighted BCE Loss`. Replace its source with:

```python
# ============================================================
# 2.2  Loss — focal loss with label smoothing (replaces BCE pos_weight)
# ============================================================
from training import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05).to(DEVICE)
print(f"Loss: FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)")
```

- [ ] **Step 2: Modify Trainer's scheduler call (cell 26)**

In cell 26 (`class Trainer:` definition), find the `fit` method's scheduler step:

```python
if self.scheduler:
    self.scheduler.step(val_auc)
```

Replace with:

```python
if self.scheduler is not None:
    # CosineAnnealingLR doesn't take a metric; ReduceLROnPlateau does.
    try:
        self.scheduler.step(val_auc)
    except TypeError:
        self.scheduler.step()
```

- [ ] **Step 3: Replace Phase 2 scheduler in cell 28**

In cell 28 (`# 4.4  Train DenseNet-121 — Phase 2`), find:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_phase2, mode="max", factor=0.5, patience=3
)
```

Replace with:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=30)
```

- [ ] **Step 4: Replace Phase 2 scheduler in cell 30 (EfficientNet)**

In cell 30 (`# 4.5  Train EfficientNet-B4`), find:

```python
scheduler_eff = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_eff_p2, mode="max", factor=0.5, patience=3, verbose=True
)
```

Replace with:

```python
scheduler_eff = optim.lr_scheduler.CosineAnnealingLR(optimizer_eff_p2, T_max=30)
```

Also in cell 30, find `epochs=20` in the Phase 2 fit call and change to `epochs=30`. Increase trainer patience from 5 to 7 (change `patience=5` to `patience=7` in the trainer instantiation).

- [ ] **Step 5: Commit**

```bash
git add "AI Healthcare/project.ipynb"
git commit -m "notebook: switch loss to FocalLoss and Phase-2 scheduler to CosineAnnealingLR"
```

---

## Task 13: Notebook — switch Grad-CAM to GradCAM++ on `blocks[-1]`

**Files:**
- Modify: `AI Healthcare/project.ipynb`

- [ ] **Step 1: Replace cell 33 (Grad-CAM helpers)**

Find the cell starting `# 5.1  Grad-CAM implementation`. Replace its source with:

```python
# ============================================================
# 5.1  Grad-CAM — delegated to backend/grad_cam_utils.py
# ============================================================
from grad_cam_utils import (
    bbox_from_heatmap_percentile,
    gradcam_pp_heatmap,
    target_layer_for_efficientnet_b4,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_target_layer(model_wrapper):
    if model_wrapper.backbone_name == "densenet121":
        return model_wrapper.model.features.denseblock4.denselayer16.norm2
    elif model_wrapper.backbone_name == "efficientnet_b4":
        return target_layer_for_efficientnet_b4(model_wrapper)
    raise ValueError("Unknown backbone")

def generate_gradcam(model_wrapper, image_tensor, target_layer):
    return gradcam_pp_heatmap(model_wrapper, image_tensor, target_layer)
```

- [ ] **Step 2: Run cell 34 (visualize_gradcam_samples) and inspect output**

After loading weights (cell 31), run cell 34. The Grad-CAM hotspots on true-positive cases should fall **inside** the lung silhouette of the displayed (lung-ROI-cropped) image, not on the edges.

If hotspots are still on the edges: this is the regression we care about. Capture a screenshot, fail the task, and report back — likely indicates the lung_roi_crop is letting too much border through and we need to tune the fallback inset.

- [ ] **Step 3: Commit**

```bash
git add "AI Healthcare/project.ipynb"
git commit -m "notebook: switch Grad-CAM to GradCAM++ on EfficientNet blocks[-1]"
```

---

## Task 14: Notebook — persist threshold + img_size into checkpoint

**Files:**
- Modify: `AI Healthcare/project.ipynb`

- [ ] **Step 1: Add TTA-aware validation cell after cell 31 (load weights)**

Insert a new cell after cell 31:

```python
# ============================================================
# 5.0  Compute TTA-aware val probs + pick the clinical threshold
# ============================================================
from cxr_pipeline import tta_predict
from training import threshold_at_specificity

model_eff.eval()
val_probs, val_labels = [], []
for i in range(len(val_ds)):
    x, y = val_ds[i]
    x = x.unsqueeze(0).to(DEVICE)
    val_probs.append(tta_predict(model_eff, x))
    val_labels.append(float(y))
val_probs = np.array(val_probs)
val_labels = np.array(val_labels)

VAL_THRESHOLD_95SPEC = threshold_at_specificity(val_labels, val_probs, target_spec=0.95)
print(f"Val threshold @ 95% spec (with TTA): {VAL_THRESHOLD_95SPEC:.4f}")
```

- [ ] **Step 2: Replace cell 45 (save final model) entirely**

Find the cell starting `# 9.1  Save final model in multiple formats`. Replace its source with:

```python
# ============================================================
# 9.1  Save final model with embedded threshold + img_size
# ============================================================
torch.save({
    "model_state_dict": model_eff.state_dict(),
    "model_name": "efficientnet_b4",
    "img_size": IMG_SIZE,
    "threshold": float(VAL_THRESHOLD_95SPEC),
    "metrics": {
        "auc": results_eff["auc_mean"],
        "sensitivity_at_spec95": results_eff["sensitivity_at_spec95"],
        "f1": results_eff["f1_score"],
    },
}, OUTPUT_DIR / "best_model.pt")

print(f"Saved best_model.pt with threshold={VAL_THRESHOLD_95SPEC:.4f}, img_size={IMG_SIZE}")
```

- [ ] **Step 3: Commit**

```bash
git add "AI Healthcare/project.ipynb"
git commit -m "notebook: persist clinical threshold + img_size in checkpoint"
```

---

## Task 15: Move legacy checkpoint out of the way + smoke-test the backend with mock weights

**Files:**
- Move: `AI Healthcare/outputs/best_model.pt` → `AI Healthcare/outputs/legacy/best_model.pt`

- [ ] **Step 1: Move the legacy checkpoint**

```bash
cd "/home/joao/Documents/last-sem/.claude/worktrees/cxr-pipeline-overhaul/AI Healthcare"
mkdir -p outputs/legacy
git mv outputs/best_model.pt outputs/legacy/best_model.pt
```

- [ ] **Step 2: Start the backend and call /health**

In one terminal:

```bash
cd "/home/joao/Documents/last-sem/.claude/worktrees/cxr-pipeline-overhaul/AI Healthcare"
source venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl -s http://localhost:8000/health
kill %1
```

Expected JSON contains `"weights_loaded": false` (legacy weights no longer at the expected path) and `"inference_size": 640`.

- [ ] **Step 3: Smoke-test /predict with mock weights and one DICOM**

```bash
cd "/home/joao/Documents/last-sem/.claude/worktrees/cxr-pipeline-overhaul/AI Healthcare/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 3
SAMPLE=$(ls "../rsna-pneumonia-detection-challenge/stage_2_train_images/" | head -1)
curl -s -X POST http://localhost:8000/predict \
     -F "file=@../rsna-pneumonia-detection-challenge/stage_2_train_images/${SAMPLE}" \
     | python -c "import json,sys; r=json.load(sys.stdin); print('mock=',r['mock'],'bbox=',r['bbox'])"
kill %1
```

Expected: `mock= True bbox= {'x':...,'y':...,'width':...,'height':...}` (mock response served because no weights).

- [ ] **Step 4: Commit**

```bash
cd "/home/joao/Documents/last-sem/.claude/worktrees/cxr-pipeline-overhaul"
git add "AI Healthcare/outputs"
git commit -m "chore: stash legacy checkpoint under outputs/legacy/ pending retrain"
```

---

## Task 16: Retrain end-to-end and validate Grad-CAM stays inside lungs

**Files:**
- Run: `AI Healthcare/project.ipynb` cells 0 through 45

- [ ] **Step 1: Run the notebook top to bottom**

Restart kernel, "Run all". This includes:
- Cells 1–14: dataset load, EDA (already validated)
- Cell 14.5 (cohort filter)
- Cells 15–19: split + datasets + DataLoaders
- Cells 20–23: baseline logistic regression
- Cells 25–30: train DenseNet (Phase 1+2) AND EfficientNet-B4 (Phase 1+2), each with FocalLoss + cosine LR
- Cell 31: load best weights into `model_eff`
- New cell 31b: TTA-aware threshold computation
- Cells 33–34: Grad-CAM visualization
- Cells 36–41: evaluation
- Cell 45: save checkpoint with threshold

Expected total runtime: 2–4 hours on a single modern GPU; longer on CPU.

Watch for:
- Disk: each `best_model.pt` is ~70 MB. Monitor with `df -h /home`.
- Memory: `BATCH_SIZE=16` at 640 might OOM on smaller GPUs — reduce to 8 if needed.

- [ ] **Step 2: Inspect Grad-CAM output qualitatively**

Open `outputs/gradcam_efficientnet.png` (saved by cell 34). For the 4 pneumonia panels, the red hotspots should overlay lung tissue, not the image edges.

If they still hug the borders → see Task 13 Step 2 (likely a tuning issue with `lung_roi_crop` or a class-imbalance regression). Stop, debug, redo.

- [ ] **Step 3: Confirm checkpoint has threshold**

```bash
cd "/home/joao/Documents/last-sem/.claude/worktrees/cxr-pipeline-overhaul/AI Healthcare"
python -c "import torch; c=torch.load('outputs/best_model.pt', map_location='cpu', weights_only=False); print({k:c[k] for k in ['model_name','img_size','threshold']})"
```

Expected: prints `{'model_name': 'efficientnet_b4', 'img_size': 640, 'threshold': <float between 0.05 and 0.6>}`.

- [ ] **Step 4: Commit notebook outputs + new checkpoint**

```bash
git add "AI Healthcare/project.ipynb" "AI Healthcare/outputs/"
git commit -m "training: retrain EfficientNet-B4 with focal loss + cosine LR + lung-ROI crop"
```

---

## Task 17: End-to-end smoke + border-attention regression test

**Files:**
- Create: `AI Healthcare/backend/tests/test_e2e_smoke.py`

- [ ] **Step 1: Write smoke test**

Create `AI Healthcare/backend/tests/test_e2e_smoke.py`:

```python
"""
End-to-end smoke: serve a real RSNA DICOM through the FastAPI app's predict
endpoint and assert the response is well-formed. Skipped if no trained weights.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATASET = Path(__file__).resolve().parents[2] / "rsna-pneumonia-detection-challenge" / "stage_2_train_images"
CKPT = Path(__file__).resolve().parents[2] / "outputs" / "best_model.pt"


@pytest.fixture(scope="module")
def client():
    if not CKPT.exists():
        pytest.skip(f"no trained checkpoint at {CKPT}")
    os.chdir(str(Path(__file__).resolve().parents[1]))  # backend/ as cwd
    from main import app
    return TestClient(app)


def test_predict_returns_centered_bbox(client):
    sample = next(DATASET.glob("*.dcm"))
    with sample.open("rb") as f:
        resp = client.post("/predict", files={"file": (sample.name, f, "application/dicom")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["weights_loaded"] is True
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    if data["prediction"] == "pneumonia":
        bbox = data["bbox"]
        assert bbox is not None
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"] / 2
        # Centroid must be at least 10% away from any edge (in 512-space)
        assert 51 <= cx <= 461, f"bbox centroid x={cx} too close to edge"
        assert 51 <= cy <= 461, f"bbox centroid y={cy} too close to edge"
```

- [ ] **Step 2: Run the smoke test**

Run from `AI Healthcare/`:

```bash
pytest backend/tests/test_e2e_smoke.py -v
```

Expected: PASS (or SKIP with clear message if no checkpoint).

- [ ] **Step 3: Run the full test suite as a final sanity check**

Run from `AI Healthcare/`:

```bash
pytest backend/tests -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add "AI Healthcare/backend/tests/test_e2e_smoke.py"
git commit -m "test: end-to-end smoke covering border-attention regression"
```

---

## Done state

- New backend modules: `cxr_pipeline.py`, `training.py`, `grad_cam_utils.py`, each with green unit tests.
- `main.py` no longer holds `_crop_borders`, `BORDER_CROP_FRACTION`, or `BBOX_BORDER_SUPPRESSION` — preprocessing is delegated, threshold comes from the checkpoint.
- Notebook reads from `cxr_pipeline` / `training` / `grad_cam_utils` and trains EfficientNet-B4 with FocalLoss + cosine LR on the cohort-filtered split.
- `outputs/best_model.pt` contains `state_dict`, `img_size=640`, `threshold` (val-set 95%-spec).
- E2E smoke asserts Grad-CAM bbox is interior.

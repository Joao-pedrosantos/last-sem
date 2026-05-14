# CXR Pipeline Overhaul — Design

**Date:** 2026-05-14
**Scope:** Backend training + serving pipeline for the RSNA Pneumonia Detection model. Frontend untouched.

## Problem

The current model's Grad-CAM lights up image borders rather than lung fields. Symptom of shortcut learning: the model is exploiting cohort-specific edge pixels (burn-in laterality markers, collimator margins, date stamps) instead of pulmonary opacities. Inference-side hacks in `backend/main.py` (`_crop_borders`, `BBOX_BORDER_SUPPRESSION`) patch the visualization but cannot fix what the model learned during training.

## Root causes identified in current code

1. **No lung-region crop, ever.** `RSNAPneumoniaDataset.__getitem__` does `Resize(512, 512)` on a raw 1024×1024 DICOM. Borders, markers, and edge artifacts go straight to the model.
2. **Per-image min/max normalization is dominated by burn-in.** `(img - min)/(max - min)` lets a single bright marker rescale the histogram — lungs squeezed into a narrow band.
3. **No DICOM VOI LUT / MONOCHROME1 handling.** Some RSNA DICOMs have inverted polarity and reach the model with reversed contrast.
4. **Grad-CAM target layer `bn2` / `conv_head`** gives only a 16×16 attention map — coarse, edge-prone.
5. **Train/inference preprocessing diverged.** Backend now does a 6% inset crop the training never saw; inference distribution shifted from train distribution.
6. **"No Lung Opacity / Not Normal" folded into negatives.** Contains other pathologies (cardiomegaly, effusion) that look like pneumonia, pushing the model toward non-pathological cues.
7. **5-epoch frozen phase + BCE with `pos_weight`.** Combined with 1–6, the model converges fast onto a shortcut.

## Architecture

One new module — `backend/cxr_pipeline.py` — owns *all* image transformations. Imported by both the training notebook and the serving backend.

```
cxr_pipeline.py
├── read_dicom(file_bytes_or_path)      → uint8 H×W grayscale, lung-bright polarity
├── lung_roi_crop(img_gray)             → uint8 H×W cropped to lung bounding box
├── to_rgb(img_gray, size)              → uint8 H×W×3 ready for transforms
├── make_train_transforms(size)         → albumentations.Compose
├── make_val_transforms(size)           → albumentations.Compose
└── tta_predict(model, tensor)          → float (mean of original + hflip sigmoids)
```

**The load-bearing decision:** one preprocessing function, one source of truth. Train and serve cannot drift.

Consequences:
- `backend/preprocessing.py` shrinks to a thin compatibility layer (or is removed and its callers re-pointed).
- `backend/main.py` loses `_crop_borders`, `BORDER_CROP_FRACTION`, `BBOX_BORDER_SUPPRESSION` — they become unnecessary.
- The notebook `RSNAPneumoniaDataset` calls `read_dicom` + `lung_roi_crop` + `to_rgb` instead of inlining its own logic.

## Components

### 1. `read_dicom(file_bytes_or_path) -> np.ndarray`

Returns uint8 H×W grayscale with lung tissue bright.

Pixel-mapping precedence:
1. If the DICOM has a VOI LUT, apply `pydicom.pixels.apply_voi_lut`.
2. Else if `WindowCenter` and `WindowWidth` are present, apply windowing.
3. Else clip to 1st–99th percentile of pixel values and linearly rescale to [0, 255].

After mapping, if `PhotometricInterpretation == "MONOCHROME1"`, invert (`255 - x`).

### 2. `lung_roi_crop(img_gray) -> np.ndarray`

Returns the input cropped to the lung bounding box, padded to square.

Algorithm:
1. Downsample to 256×256 for speed.
2. Otsu threshold → binary mask.
3. Keep largest connected component; morphologically close (3×3 kernel, 2 iters) to fill the mediastinum.
4. Bounding box of the resulting blob.
5. Sanity check: bbox must cover ≥35% of image area *and* its aspect ratio must be in [0.5, 2.0]. If not, fall back to a fixed 8% inset (`img[h*0.08:h*0.92, w*0.08:w*0.92]`).
6. Map bbox back to the original resolution and crop.
7. Pad shorter side with zeros to make the crop square (preserves aspect ratio under later resize).

No exceptions raised — every input produces *some* output, deterministic per input.

### 3. `to_rgb(img_gray, size) -> np.ndarray`

`PIL.Image.resize((size, size), Image.BILINEAR)` then `np.stack([gray]*3, axis=-1)`.

### 4. Augmentations

Training (`make_train_transforms`):
- `HorizontalFlip(p=0.5)`
- `ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5)`
- `RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)`
- `RandomGamma(gamma_limit=(80, 120), p=0.3)`
- `CoarseDropout(max_holes=4, max_height=64, max_width=64, p=0.3)` — sets random patches to zero; combined with lung crop, eliminates any residual border shortcut.
- `Normalize(mean=ImageNet, std=ImageNet)`
- `ToTensorV2`

Validation / inference (`make_val_transforms`):
- `Normalize(mean=ImageNet, std=ImageNet)`
- `ToTensorV2`

(Resize is already done by `to_rgb` so no `A.Resize` in either pipeline.)

### 5. Training recipe (notebook)

- **Cohort filter:** drop "No Lung Opacity / Not Normal" patients from train and val splits. Keep them as a *separate stress-eval bucket* on the test set, reported but not used for early stopping.
- **Loss:** `FocalLoss(alpha=0.25, gamma=2.0)` with label smoothing 0.05. Replaces `BCEWithLogitsLoss(pos_weight=...)`.
- **Phase 1 (head only):** lr=1e-4, 5 epochs, frozen backbone.
- **Phase 2 (fine-tune last 2 blocks):** lr=1e-5, **cosine LR schedule** (`CosineAnnealingLR(T_max=30)`), 30 epochs, early stop on val AUC with patience=7.
- **Input size:** **640×640** during training and inference (EfficientNet-B4's native train size, well-established RSNA setting).
- **TTA at val/test:** average sigmoid of original + horizontal flip. Used for the threshold-selection metric only; not in the train loop.

### 6. Operating threshold persistence

After training, compute the threshold at 95% specificity on the validation set (with TTA). Store it in the checkpoint:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": "efficientnet_b4",
    "img_size": 640,
    "threshold": threshold_at_95_spec,
    "metrics": { ... },
}, OUTPUT_DIR / "best_model.pt")
```

The backend loads `checkpoint["threshold"]` if present; falls back to env var (`CLINICAL_THRESHOLD`) only if missing. The env var is no longer the primary source.

### 7. Grad-CAM upgrade

- **Target layer:** `model.model.blocks[-1]` (last MBConv stage) — richer features than `bn2`.
- **Algorithm:** `GradCAMPlusPlus` from `pytorch-grad-cam`.
- **Post-processing:** Gaussian blur (σ=8 in 512-display space) on the heatmap before bbox extraction.
- **Bbox extraction:** threshold at the **90th percentile** of heatmap values (not 50% of max). Yields a tighter, more clinically-plausible region.

The bbox is only returned when `prediction == "pneumonia"` (unchanged from current behavior).

### 8. `tta_predict(model, tensor) -> float`

```python
with torch.no_grad():
    p1 = sigmoid(model(tensor)).item()
    p2 = sigmoid(model(torch.flip(tensor, dims=[-1]))).item()
return (p1 + p2) / 2
```

Used by the serving backend on every prediction. Doubles inference cost; acceptable for ER-triage latency budget.

## Data flow

```
DICOM bytes
    │
    ▼
read_dicom ───────► uint8 H×W (lung-bright, polarity-normalized)
    │
    ▼
lung_roi_crop ───► uint8 H×W cropped+squared
    │
    ├──► display_image (resized to 512×512 for the frontend canvas + bbox space)
    │
    ▼
to_rgb(size=640) ─► uint8 640×640×3
    │
    ▼
val_transforms ──► tensor (1, 3, 640, 640)
    │
    ▼
tta_predict ─────► probability
    │
    ▼
sigmoid + threshold → prediction
    │
    ▼
GradCAM++ on blocks[-1] → heatmap (640×640) → resize to 512 → blur → bbox (in 512 space)
```

The display image and the bbox both live in 512-space so the frontend overlay is pixel-aligned.

## Error handling

| Failure | Behavior |
|---|---|
| DICOM read fails (corrupt file) | HTTP 422 with message (current behavior preserved). |
| Otsu produces degenerate bbox | Silent fallback to fixed 8% inset crop. |
| Model weights file missing | Existing `mock=true` placeholder path. Unchanged. |
| Grad-CAM raises | Return prediction without `gradcam`/`bbox` (current behavior preserved). |
| Checkpoint missing `threshold` key | Fall back to `CLINICAL_THRESHOLD` env var, then 0.5. |

## Testing

**Unit (`tests/test_cxr_pipeline.py`):**
- `read_dicom` correctly inverts a synthetic MONOCHROME1 DICOM (built with `pydicom.Dataset`).
- `read_dicom` applies windowing when only WC/WW present.
- `lung_roi_crop` on a synthetic image with a 100-px black border returns a bbox within ±5px of the inner rectangle.
- `lung_roi_crop` on a uniform-gray image falls back to the fixed 8% inset.
- `lung_roi_crop` on an image where the largest blob is unrealistically thin (aspect > 2.0) falls back.

**Integration:**
- After retrain, on the held-out test set: assert that the Grad-CAM 90th-percentile centroid is **inside** the lung-ROI crop bbox for ≥90% of true-positive predictions. (Border-attention regression test.)
- Sens@Spec95 on val set ≥ 0.50 (current notebook's baseline floor — anything worse is a real regression).

**Smoke:**
- `POST /predict` with a known RSNA-positive DICOM returns a bbox whose centroid is not within 10% of any image edge.

## Migration

- Existing `outputs/best_model.pt` is left in place under `outputs/legacy/best_model.pt` for comparison plots only; the backend points at the new `outputs/best_model.pt` after retrain.
- The new checkpoint contains `threshold` and `img_size`. The backend reads both. If `img_size` is missing, defaults to 640.

## Out of scope

- U-Net lung segmentation (deferred; only revisit if Tier A+B+C plateaus below sens@spec95 = 0.65).
- ONNX export, model ensembling.
- Backbone swap.
- Multi-class (3-class) classification.

## Open items pre-implementation

None blocking. Dataset is present at `AI Healthcare/rsna-pneumonia-detection-challenge/` with `stage_2_train_labels.csv` extracted.

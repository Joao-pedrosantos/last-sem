# CXR Pneumonia Detection

EfficientNet-B4 binary classifier on the RSNA Pneumonia Detection Challenge.
Backend serves Grad-CAM++ heatmaps over a FastAPI `/predict` endpoint;
training is driven by `project.ipynb`.

## Quick start on a new machine

```bash
git clone <repo> last-sem
cd last-sem/AI\ Healthcare
./setup_home.sh                  # creates ./venv with the right CUDA wheel
```

`setup_home.sh` auto-detects an NVIDIA GPU and installs the `cu124` PyTorch
wheel (which covers RTX 50-series / Blackwell, SM_120). Override with
`CUDA=cu126 ./setup_home.sh` for nightly, or `CUDA=cpu` for CPU-only.

## Train

1. Drop your Kaggle API token at `~/.kaggle/kaggle.json` (chmod 600).
2. Launch Jupyter inside the venv and run `project.ipynb` from the top:

   ```bash
   source venv/bin/activate
   jupyter lab project.ipynb
   ```

3. Cell 0.4 downloads the RSNA dataset (~3.4 GB) into
   `./rsna-pneumonia-detection-challenge/`. After it finishes, the inner
   labels archive is auto-extracted by the loading cells (or you can unzip
   `stage_2_train_labels.csv.zip` manually).

4. "Run all" trains EfficientNet-B4 with focal loss + cosine LR schedule
   (Phase 1: 5 epochs frozen, Phase 2: 30 epochs fine-tuned). The final
   cell writes `outputs/best_model.pt` containing
   `{model_state_dict, model_name, img_size, threshold, metrics}` — the
   clinical operating threshold is picked at 95% specificity on the
   validation set with test-time augmentation.

   Wall-clock on an RTX 5070 Ti (16 GB): roughly 30–45 min for the full
   protocol at 640×640, batch 16.

## Serve

```bash
./start.sh                       # localhost:8000
```

`/health` reports `weights_loaded` + the active threshold and inference
size.  `/predict` accepts DICOM, PNG, or JPEG and returns prediction,
probability, base64 lung-cropped display image, Grad-CAM++ overlay, and
a bounding box (90th-percentile envelope of the heatmap).

## Architecture

The serving and training paths share one preprocessing module:

```
backend/cxr_pipeline.py      DICOM read (VOI LUT + MONOCHROME1)
                             Otsu lung-ROI crop with 8% inset fallback
                             to_rgb + albumentations transforms
                             tta_predict (original + hflip avg)

backend/training.py          FocalLoss + threshold_at_specificity

backend/grad_cam_utils.py    GradCAM++ + 90th-percentile bbox extractor
```

The notebook imports these from `backend/` so training and serving
cannot drift.

## Tests

```bash
source venv/bin/activate
cd AI\ Healthcare
pytest backend/tests -v
```

16 unit tests cover DICOM polarity handling, lung-crop fallbacks,
transform shape/dtype, TTA averaging, focal loss reduction, threshold
selection, and bbox extraction.

## What changed from the previous model

The previous model's Grad-CAM hugged the image edges — the classifier
had latched onto burn-in markers and collimator borders instead of
lung opacities. Fixes baked into this rewrite:

1. **Lung-ROI crop before model input** (Otsu + connected components).
2. **VOI LUT + MONOCHROME1 polarity handling** in DICOM read.
3. **CoarseDropout augmentation** to penalize edge-pixel reliance.
4. **Cohort filter**: "No Lung Opacity / Not Normal" patients moved
   out of train/val into a stress-eval bucket.
5. **Focal loss** replaces BCE-with-pos-weight for the imbalanced
   binary problem; label smoothing 0.05.
6. **Grad-CAM++** on EfficientNet's last MBConv block (richer than
   `bn2`), 90th-percentile bbox extraction with Gaussian blur.
7. **Operating threshold persisted in the checkpoint** so the
   backend doesn't have to guess.

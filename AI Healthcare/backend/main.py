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
    lung_focused_crop,
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
FALLBACK_THRESHOLD = float(os.getenv("CLINICAL_THRESHOLD", "0.5"))

DISPLAY_SIZE = 512
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


def _is_dicom(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in DICOM_EXTENSIONS


def _array_to_base64_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _display_image(img_gray: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img_gray).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    arr = np.array(pil)
    return np.stack([arr, arr, arr], axis=-1)


def _mock_response() -> dict:
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

    try:
        if _is_dicom(file.filename):
            img_gray = read_dicom(file_bytes)
        else:
            img_rgb = _load_standard_image(file_bytes)
            img_gray = np.array(Image.fromarray(img_rgb).convert("L"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc

    img_cropped, lung_mask_crop = lung_focused_crop(img_gray)
    display_rgb = _display_image(img_cropped)

    img_rgb_model = to_rgb(img_cropped, size=INFERENCE_SIZE)
    tensor = make_val_transforms(INFERENCE_SIZE)(image=img_rgb_model)["image"].unsqueeze(0).to(DEVICE)

    probability = tta_predict(model, tensor)
    prediction = "pneumonia" if probability >= CLINICAL_THRESHOLD else "normal"

    gradcam_b64 = None
    bbox = None
    try:
        heatmap = gradcam_pp_heatmap(
            model_wrapper=model,
            tensor=tensor,
            target_layer=target_layer_for_efficientnet_b4(model),
        )
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import Image as PILImage
        # Resize lung mask to display size — used to gate both the heatmap and
        # the bbox so neither lights up outside the lung field.
        mask_disp = np.array(
            PILImage.fromarray(lung_mask_crop.astype(np.uint8) * 255)
            .resize((DISPLAY_SIZE, DISPLAY_SIZE), PILImage.NEAREST)
        ) > 127
        heatmap_disp = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8))
            .resize((DISPLAY_SIZE, DISPLAY_SIZE), PILImage.BILINEAR)
        ).astype(np.float32) / 255.0
        heatmap_disp = heatmap_disp * mask_disp.astype(np.float32)
        img_float = display_rgb.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, heatmap_disp, use_rgb=True)
        gradcam_b64 = _array_to_base64_png(cam_image)
        bbox = bbox_from_heatmap_percentile(
            heatmap, percentile=90.0, blur_sigma=8.0, size=DISPLAY_SIZE,
            lung_mask=lung_mask_crop,
        )
    except Exception:
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


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

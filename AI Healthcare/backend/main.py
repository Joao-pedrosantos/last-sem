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

from model import build_densenet, load_model
from preprocessing import load_dicom, load_image, preprocess

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
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
# Clinical operating threshold — tune after evaluating on the test set.
# Default 0.5; replace with the 95%-specificity threshold from the notebook.
CLINICAL_THRESHOLD = float(os.getenv("CLINICAL_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# App & model startup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray pneumonia classifier (DenseNet-121) — supports DICOM and standard image formats.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, weights_loaded = load_model(MODEL_PATH, DEVICE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}
DISPLAY_SIZE = 512  # all responses standardize on a 512x512 canvas
BBOX_THRESHOLD = 0.5  # fraction of heatmap max used to cut the activation region
BORDER_CROP_FRACTION = 0.06  # trim N% off each side to remove Dx/Sin markers and edge artefacts
BBOX_BORDER_SUPPRESSION = 0.08  # zero this fraction of the heatmap's border before bbox extraction


def _is_dicom(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in DICOM_EXTENSIONS


def _array_to_base64_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _crop_borders(image_array: np.ndarray, fraction: float = BORDER_CROP_FRACTION) -> np.ndarray:
    """
    Trim a fraction of pixels from each side. Chest X-rays routinely carry
    laterality markers ("Dx"/"Sin", arrows, burn-in text) and black padding in
    the corners — these are spurious features the model can latch onto. Cropping
    them out before inference keeps the model focused on the lung fields.
    """
    if fraction <= 0:
        return image_array
    h, w = image_array.shape[:2]
    dx = int(w * fraction)
    dy = int(h * fraction)
    if dx * 2 >= w or dy * 2 >= h:
        return image_array
    return image_array[dy : h - dy, dx : w - dx]


def _display_image(image_array: np.ndarray) -> np.ndarray:
    """Resize to the display canvas and ensure uint8 RGB."""
    pil = Image.fromarray(image_array)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (DISPLAY_SIZE, DISPLAY_SIZE):
        pil = pil.resize((DISPLAY_SIZE, DISPLAY_SIZE))
    return np.array(pil)


def _bbox_from_heatmap(heatmap: np.ndarray) -> dict | None:
    """
    Derive a bounding box from a Grad-CAM heatmap by thresholding at
    BBOX_THRESHOLD * max and taking the envelope of the activated pixels.
    Coordinates are returned in the 512x512 display space.
    """
    if heatmap.size == 0:
        return None

    if heatmap.shape != (DISPLAY_SIZE, DISPLAY_SIZE):
        pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE)
        )
        heatmap = np.array(pil).astype(np.float32) / 255.0

    # Suppress activations near the image border — even after cropping the
    # input, Grad-CAM on CNN features can spill into the edge receptive field.
    if BBOX_BORDER_SUPPRESSION > 0:
        b = int(DISPLAY_SIZE * BBOX_BORDER_SUPPRESSION)
        heatmap = heatmap.copy()
        heatmap[:b, :] = 0
        heatmap[-b:, :] = 0
        heatmap[:, :b] = 0
        heatmap[:, -b:] = 0

    peak = float(heatmap.max())
    if peak <= 0:
        return None

    mask = heatmap >= (peak * BBOX_THRESHOLD)
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    return {
        "x": x0,
        "y": y0,
        "width": max(1, x1 - x0),
        "height": max(1, y1 - y0),
    }


def _mock_cam_and_bbox() -> tuple[np.ndarray, np.ndarray]:
    """Placeholder image + heatmap used when weights aren't loaded."""
    h = w = DISPLAY_SIZE
    canvas = np.full((h, w, 3), 80, dtype=np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, sigma = int(w * 0.62), int(h * 0.55), 95
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    r = np.clip(canvas[:, :, 0].astype(np.int32) + (blob * 200).astype(np.int32), 0, 255).astype(np.uint8)
    g = np.clip(canvas[:, :, 1].astype(np.int32) + (blob * 80).astype(np.int32), 0, 255).astype(np.uint8)
    b = canvas[:, :, 2]

    heatmap_rgb = np.stack([r, g, b], axis=-1)
    return heatmap_rgb, blob.astype(np.float32)


def _gradcam(tensor: torch.Tensor, display_rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """
    Run Grad-CAM and return (base64 heatmap PNG, raw grayscale heatmap).
    Returns (None, None) if Grad-CAM fails.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # EfficientNet-B4 (timm): the last batch-norm after conv_head gives a
        # clean feature map for Grad-CAM. The classifier wrapper stores the
        # actual model under `.model`.
        target_layer = model.model.bn2
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W) in [0, 1]

        img_float = display_rgb.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        return _array_to_base64_png(cam_image), grayscale_cam
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Routes — must be declared before mounting static files
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "weights_loaded": weights_loaded,
        "model_path": MODEL_PATH,
        "threshold": CLINICAL_THRESHOLD,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True, description="Include Grad-CAM heatmap in response"),
):
    """
    Accept a chest X-ray (DICOM or PNG/JPEG) and return a pneumonia prediction.

    When no model weights are found the endpoint returns a **placeholder response**
    (mock=true) so the frontend can be tested end-to-end without a trained model.

    Response fields:
    - **prediction**: "pneumonia" | "normal"
    - **probability**: sigmoid probability ∈ [0, 1]
    - **threshold**: operating threshold used for the binary decision
    - **weights_loaded**: false → placeholder response, results are not meaningful
    - **mock**: true when the response is placeholder data
    - **gradcam**: base64-encoded PNG heatmap
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # ------------------------------------------------------------------ mock
    if not weights_loaded:
        mock_rgb, mock_heatmap = _mock_cam_and_bbox()
        response: dict = {
            "prediction": "pneumonia",
            "probability": 0.78,
            "threshold": CLINICAL_THRESHOLD,
            "weights_loaded": False,
            "mock": True,
            "image": {
                "base64": _array_to_base64_png(mock_rgb),
                "width": DISPLAY_SIZE,
                "height": DISPLAY_SIZE,
            },
            "bbox": _bbox_from_heatmap(mock_heatmap),
        }
        if include_gradcam:
            response["gradcam"] = _array_to_base64_png(mock_rgb)
        return response

    # ------------------------------------------------------------------ real
    try:
        if _is_dicom(file.filename):
            image_array = load_dicom(file_bytes)
        else:
            image_array = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc

    # Crop the border markers/padding out before anything else so both the
    # model and the displayed image see the same field of view and bbox
    # coordinates map 1:1 to what the user sees.
    image_array = _crop_borders(image_array)

    display_rgb = _display_image(image_array)
    tensor = preprocess(image_array).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    prediction = "pneumonia" if probability >= CLINICAL_THRESHOLD else "normal"

    gradcam_b64, heatmap = _gradcam(tensor, display_rgb)
    bbox = _bbox_from_heatmap(heatmap) if heatmap is not None else None

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
# Serve frontend — must come after API routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

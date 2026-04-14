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
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pt")
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


def _is_dicom(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in DICOM_EXTENSIONS


def _mock_gradcam() -> str:
    """
    Generate a placeholder heatmap image for layout testing.
    Simulates a Grad-CAM blob in the right lower lobe region.
    """
    h, w = 512, 512

    # Gray base (chest X-ray look)
    canvas = np.full((h, w, 3), 80, dtype=np.uint8)

    # Soft circular blob
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, sigma = int(w * 0.62), int(h * 0.55), 95
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    r = np.clip(canvas[:, :, 0].astype(np.int32) + (blob * 200).astype(np.int32), 0, 255).astype(np.uint8)
    g = np.clip(canvas[:, :, 1].astype(np.int32) + (blob * 80).astype(np.int32),  0, 255).astype(np.uint8)
    b = canvas[:, :, 2]

    heatmap = np.stack([r, g, b], axis=-1)

    buffer = io.BytesIO()
    Image.fromarray(heatmap).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _gradcam(image_array: np.ndarray, tensor: torch.Tensor) -> str | None:
    """
    Run Grad-CAM on the DenseNet-121 target layer and return a base64-encoded PNG.
    Returns None if Grad-CAM fails for any reason.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.features.denseblock4.denselayer16.norm2
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W)

        img_float = image_array.astype(np.float32) / 255.0
        if img_float.shape[:2] != (512, 512):
            pil_tmp = Image.fromarray((img_float * 255).astype(np.uint8)).resize((512, 512))
            img_float = np.array(pil_tmp).astype(np.float32) / 255.0
        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=-1)

        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        buffer = io.BytesIO()
        Image.fromarray(cam_image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None


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
    # When no trained weights exist we skip real inference entirely and return
    # a fixed placeholder so the UI can be tested end-to-end.
    if not weights_loaded:
        response: dict = {
            "prediction": "pneumonia",
            "probability": 0.78,
            "threshold": CLINICAL_THRESHOLD,
            "weights_loaded": False,
            "mock": True,
        }
        if include_gradcam:
            response["gradcam"] = _mock_gradcam()
        return response

    # ------------------------------------------------------------------ real
    try:
        if _is_dicom(file.filename):
            image_array = load_dicom(file_bytes)
        else:
            image_array = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc

    tensor = preprocess(image_array).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    prediction = "pneumonia" if probability >= CLINICAL_THRESHOLD else "normal"

    response = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": CLINICAL_THRESHOLD,
        "weights_loaded": True,
        "mock": False,
    }

    if include_gradcam:
        heatmap = _gradcam(image_array, tensor)
        response["gradcam"] = heatmap

    return response


# ---------------------------------------------------------------------------
# Serve frontend — must come after API routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

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

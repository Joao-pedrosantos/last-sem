import io
import os
import tempfile

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ImageNet normalization — matches project.ipynb
TRANSFORMS = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def load_dicom(file_bytes: bytes) -> np.ndarray:
    """
    Read a DICOM file and return a uint8 3-channel numpy array (H, W, 3).
    Pixel values are normalized to [0, 255].
    """
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError("pydicom is required to process DICOM files. Install it with: pip install pydicom") from e

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        dcm = pydicom.dcmread(tmp_path)
        pixels = dcm.pixel_array.astype(np.float32)

        # Normalize to 0-255
        pmin, pmax = pixels.min(), pixels.max()
        if pmax > pmin:
            pixels = (pixels - pmin) / (pmax - pmin) * 255.0
        pixels = pixels.astype(np.uint8)

        # Grayscale → 3-channel RGB (replicate channels for ImageNet pretrained weights)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)

        return pixels
    finally:
        os.unlink(tmp_path)


def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Read a standard image file (PNG, JPEG, …) and return a uint8 RGB numpy array.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Apply the inference transform pipeline and return a (1, 3, 512, 512) tensor.
    """
    augmented = TRANSFORMS(image=image)
    return augmented["image"].unsqueeze(0)  # add batch dimension

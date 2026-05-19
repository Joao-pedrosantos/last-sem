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
    lung_mask: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    Convert a (H, W) heatmap in [0, 1] to a bounding box dict in `size`-space.
    Returns None if the heatmap has no signal (all zeros).

    If `lung_mask` is provided it is resized to `size`×`size` (nearest neighbour
    to keep it binary) and multiplied into the heatmap *before* thresholding —
    this stops the bbox from latching onto activations outside the lung field.
    """
    if heatmap.size == 0:
        return None

    if heatmap.shape != (size, size):
        pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((size, size), Image.BILINEAR)
        heatmap = np.array(pil).astype(np.float32) / 255.0

    if lung_mask is not None:
        m = lung_mask
        if m.shape != (size, size):
            m_pil = Image.fromarray((m > 0).astype(np.uint8) * 255)
            m_pil = m_pil.resize((size, size), Image.NEAREST)
            m = (np.array(m_pil) > 127).astype(np.float32)
        else:
            m = (m > 0).astype(np.float32)
        heatmap = heatmap * m

    if blur_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

    threshold = float(np.percentile(heatmap, percentile))
    if threshold <= 0:
        return None

    mask = heatmap > threshold
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

    We pass `BinaryClassifierOutputTarget(1)` explicitly: the model emits a
    single pneumonia logit, so the relevant gradient is w.r.t. that one
    output. Letting pytorch-grad-cam auto-infer the target trips a bug for
    single-logit models (`numpy.int64 not iterable`).
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    cam = GradCAMPlusPlus(model=model_wrapper, target_layers=[target_layer])
    targets = [BinaryClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    return grayscale_cam[0]

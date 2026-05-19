import os
from typing import Tuple, Dict, Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PneumoniaClassifierMT(PneumoniaClassifier):
    """
    Same parameters as `PneumoniaClassifier` — its state_dict loads here
    unchanged. By default, `forward(x)` still returns the logit (drop-in
    replacement). Pass `return_cam=True` to also get the spatial class
    activation map.

    The CAM is the linear classifier reinterpreted as a 1×1 conv on the
    backbone feature map. GAP(cam) == original logit, so the math is
    unchanged; we just expose the spatial intermediate for either
    visualization (serving) or supervised localization (training).
    """

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        if not return_cam:
            return self.model(x)
        feat = self.model.forward_features(x)  # (B, C, h, w)
        classifier = self.model.get_classifier()
        w = classifier.weight.unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)
        b = classifier.bias  # (1,)
        cam = F.conv2d(feat, w, bias=b)  # (B, 1, h, w)
        logit = F.adaptive_avg_pool2d(cam, 1).flatten(1).squeeze(-1)  # (B,)
        return logit, cam


def load_model(weights_path: str, device: torch.device) -> Tuple[nn.Module, bool, Dict[str, Any]]:
    """
    Build the classifier and load saved weights if available.

    Returns (model, weights_loaded, meta) where meta is a dict of any extra
    fields embedded in the checkpoint — currently `img_size` and `threshold`.
    Meta is always a dict (possibly empty); the caller picks defaults for
    missing keys.
    """
    # Use the MT class so callers can opt into the CAM via return_cam=True.
    # State_dict is identical to PneumoniaClassifier so old checkpoints load.
    model = PneumoniaClassifierMT("efficientnet_b4", pretrained=False)

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

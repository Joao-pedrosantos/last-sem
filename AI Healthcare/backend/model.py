import os
import torch
import torch.nn as nn
from torchvision import models


def build_densenet(dropout: float = 0.3) -> nn.Module:
    """
    DenseNet-121 with a single-output classifier head.
    Matches the architecture defined in project.ipynb.
    """
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )
    return model


def load_model(weights_path: str, device: torch.device) -> tuple[nn.Module, bool]:
    """
    Build DenseNet-121 and load saved weights if available.
    Returns (model, weights_loaded).
    """
    model = build_densenet()

    weights_loaded = False
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        weights_loaded = True

    model.eval()
    model.to(device)
    return model, weights_loaded

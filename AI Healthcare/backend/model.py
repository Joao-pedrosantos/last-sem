import os
import timm
import torch
import torch.nn as nn


class PneumoniaClassifier(nn.Module):
    """
    Mirror of the training wrapper in project.ipynb so the saved
    state_dict (keys prefixed with `model.`) loads cleanly.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=dropout,
        )

    def forward(self, x):
        # Keep (B, 1) shape so pytorch_grad_cam's ClassifierOutputTarget can
        # index the class; .item() still works for batch=1 inference.
        return self.model(x)


def build_densenet(dropout: float = 0.3) -> nn.Module:
    """Kept for backwards compatibility — returns the EfficientNet-B4 wrapper
    that matches the checkpoint actually saved by the notebook."""
    return PneumoniaClassifier("efficientnet_b4", pretrained=False, dropout=dropout)


def load_model(weights_path: str, device: torch.device) -> tuple[nn.Module, bool]:
    """
    Build the classifier and load saved weights if available.
    Returns (model, weights_loaded).
    """
    model = PneumoniaClassifier("efficientnet_b4", pretrained=False)

    weights_loaded = False
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        weights_loaded = True

    model.eval()
    model.to(device)
    return model, weights_loaded

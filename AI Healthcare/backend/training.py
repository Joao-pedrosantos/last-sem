"""
Training-time utilities: focal loss and clinical-threshold helper.
Used by the notebook, not the serving backend.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced single-logit classifiers.

    loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(logit) for target=1, else 1 - sigmoid(logit).

    Label smoothing pulls targets toward 0.5 by `label_smoothing` units before
    computing pt — softens overconfidence without changing class balance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.label_smoothing > 0:
            ls = self.label_smoothing
            targets = targets * (1.0 - ls) + 0.5 * ls

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


def threshold_at_specificity(y_true: np.ndarray, y_proba: np.ndarray,
                             target_spec: float = 0.95) -> float:
    """
    Return the probability threshold whose specificity is closest to `target_spec`
    on the ROC curve. Used to pick the clinical operating point.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificities = 1.0 - fpr
    # Filter out infinite thresholds to get practical operating points
    finite_mask = np.isfinite(thresholds)
    if finite_mask.any():
        finite_specs = specificities[finite_mask]
        finite_thrs = thresholds[finite_mask]
        spec_diffs = np.abs(finite_specs - target_spec)
        # Among ties in specificity distance, prefer smaller thresholds
        best_spec_diff = spec_diffs.min()
        best_indices = np.where(spec_diffs == best_spec_diff)[0]
        idx = best_indices[0] if len(best_indices) == 1 else best_indices[np.argmin(finite_thrs[best_indices])]
        return float(finite_thrs[idx])
    else:
        idx = int(np.argmin(np.abs(specificities - target_spec)))
        return float(thresholds[idx])

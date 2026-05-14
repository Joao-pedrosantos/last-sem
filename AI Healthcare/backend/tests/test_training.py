import numpy as np
import torch
import torch.nn.functional as F


def test_focal_loss_equals_bce_when_gamma_zero_and_alpha_half():
    from backend.training import FocalLoss
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    focal = FocalLoss(alpha=0.5, gamma=0.0, label_smoothing=0.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean") * 0.5
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    from backend.training import FocalLoss
    logits_easy = torch.tensor([5.0, -5.0])
    targets_easy = torch.tensor([1.0, 0.0])
    logits_hard = torch.tensor([-5.0, 5.0])
    targets_hard = torch.tensor([1.0, 0.0])

    focal = FocalLoss(alpha=0.5, gamma=2.0)
    loss_easy = focal(logits_easy, targets_easy)
    loss_hard = focal(logits_hard, targets_hard)
    assert loss_hard > 100 * loss_easy


def test_threshold_at_specificity_picks_correct_cutoff():
    from backend.training import threshold_at_specificity
    rng = np.random.default_rng(0)
    y_neg = rng.uniform(0.0, 0.5, 100)
    y_pos = rng.uniform(0.5, 1.0, 100)
    y_proba = np.concatenate([y_neg, y_pos])
    y_true = np.concatenate([np.zeros(100), np.ones(100)])

    thr = threshold_at_specificity(y_true, y_proba, target_spec=0.95)
    assert 0.4 < thr < 0.6

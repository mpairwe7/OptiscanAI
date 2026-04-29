"""
Loss functions for multi-label retinal disease classification.
Supports label smoothing, class weighting, and focal/asymmetric variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing for multi-label: push toward 0.5."""
    if smoothing <= 0:
        return targets
    return targets * (1 - smoothing) + 0.5 * smoothing


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing support.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pos_weight=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _smooth_targets(targets, self.label_smoothing)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # Per-sample alpha: higher weight for positives (rare), lower for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL, ICCV 2021) with label smoothing.
    Different focusing for positives vs negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _smooth_targets(targets, self.label_smoothing)
        probs_pos = torch.sigmoid(logits).clamp(min=1e-8, max=1 - 1e-8)
        probs_neg = 1.0 - probs_pos

        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos)
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))

        pt = probs_pos * targets + probs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = (1.0 - pt).clamp(min=0.0) ** gamma

        return -((loss_pos + loss_neg) * focal_weight).mean()


def build_loss(cfg: dict, pos_weight=None) -> nn.Module:
    """Factory: build loss from config, now reads label_smoothing."""
    train_cfg = cfg.get("training", {})
    loss_name = train_cfg.get("loss", "focal")
    smoothing = train_cfg.get("label_smoothing", 0.0)

    if loss_name == "focal":
        return FocalLoss(
            alpha=train_cfg.get("focal_alpha", 0.25),
            gamma=train_cfg.get("focal_gamma", 2.0),
            label_smoothing=smoothing,
            pos_weight=pos_weight,
        )
    elif loss_name == "asymmetric":
        return AsymmetricLoss(label_smoothing=smoothing)
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

"""
MixUp and CutMix augmentation for multi-label classification.
Applied at the batch level after collation.
"""
from __future__ import annotations

import numpy as np
import torch


class MixUpCutMix:
    """
    Batch-level MixUp and CutMix with configurable probability.
    Works with multi-label float targets (no one-hot needed).
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if np.random.random() > self.prob:
            return images, targets

        if np.random.random() < self.switch_prob and self.cutmix_alpha > 0:
            return self._cutmix(images, targets)
        return self._mixup(images, targets)

    def _mixup(self, images, targets):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        perm = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[perm]
        mixed_targets = lam * targets + (1 - lam) * targets[perm]
        return mixed_images, mixed_targets

    def _cutmix(self, images, targets):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1.0
        perm = torch.randperm(images.size(0), device=images.device)
        B, C, H, W = images.shape

        # Random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # Adjust lambda by actual area ratio
        lam_adjusted = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        mixed_targets = lam_adjusted * targets + (1 - lam_adjusted) * targets[perm]
        return mixed_images, mixed_targets


def build_mixup(cfg: dict) -> MixUpCutMix | None:
    """Build MixUp/CutMix from config. Returns None if disabled."""
    aug_root = cfg.get("augmentation", {})
    aug = aug_root.get("train", aug_root)
    mixup_alpha = aug.get("mixup_alpha", 0)
    cutmix_alpha = aug.get("cutmix_alpha", 0)
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None
    return MixUpCutMix(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=aug.get("mixup_cutmix_prob", 0.5),
    )

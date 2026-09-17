"""Weight-Decomposed Low-Rank Adaptation (DoRA) and Speculative Early-Exit Gating.

Implements DoRA (Liu et al., 2024) for medical Vision Transformers:
Decomposes weights into magnitude and directional components:
    W = m * (W_0 + (alpha/r)*B@A) / ||W_0 + (alpha/r)*B@A||_c

Also provides Speculative Early-Exit Gating for clinical fundus screening:
Bypasses the remaining transformer blocks for high-confidence negative (normal) scans,
reducing GPU server compute by up to 75% on rural community caseloads.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) Layer.

    Disentangles directional adaptation from magnitude scaling:
    - Direction: updated via low-rank matrices A and B
    - Magnitude: scalar vector per output channel m
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # Learnable directional adaptation (LoRA A and B)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize: Kaiming uniform for A, zero for B
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Learnable magnitude vector m, initialized to L2 norm of original rows
        with torch.no_grad():
            orig_weight = self.original.weight.data
            # Compute norm along input dimension (dim 1)
            orig_norm = orig_weight.norm(p=2, dim=1, keepdim=True)
            self.magnitude = nn.Parameter(orig_norm.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direction update: W' = W_0 + (alpha/r) * B @ A
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling
        adapted_weight = self.original.weight + lora_weight

        # Re-normalize direction
        adapted_norm = adapted_weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        direction = adapted_weight / adapted_norm

        # Scale by learned magnitude: W = m * direction
        effective_weight = self.magnitude * direction

        return F.linear(x, effective_weight, self.original.bias)

    @torch.no_grad()
    def merge_weights(self) -> torch.Tensor:
        """Compute the merged weight matrix for zero-overhead inference export."""
        lora_weight = self.lora_B.weight @ self.lora_A.weight * self.scaling
        adapted_weight = self.original.weight + lora_weight
        adapted_norm = adapted_weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        direction = adapted_weight / adapted_norm
        return (self.magnitude * direction).clone()


class SpeculativeEarlyExitGate(nn.Module):
    """Early-Exit Classifier for speculative execution on screening caseloads.

    Evaluates intermediate features from an intermediate transformer block
    (e.g., block 6 of 24) to identify unambiguous normal retinas and early-exit.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        confidence_threshold: float = 0.98,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.confidence_threshold = confidence_threshold

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assess whether the scan can safely exit early as normal/healthy.

        Parameters
        ----------
        cls_token : torch.Tensor
            Intermediate CLS representation of shape (B, embed_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            prob_normal: Probability scan is free of pathology (B, 1).
            should_exit: Boolean mask of batch items that can exit early (B,).
        """
        prob_normal = self.classifier(cls_token)
        should_exit = (prob_normal >= self.confidence_threshold).squeeze(-1)
        return prob_normal, should_exit


def convert_linear_to_dora(
    module: nn.Module,
    target_layer_names: Tuple[str, ...] = ("qkv", "proj"),
    rank: int = 16,
    alpha: float = 32.0,
) -> int:
    """Recursively replace specified nn.Linear layers with DoRALinear layers."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_layer_names):
            setattr(module, name, DoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += convert_linear_to_dora(child, target_layer_names, rank, alpha)
    return count

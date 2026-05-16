"""MobileStudentV1 — Lightweight student model for on-device retinal screening.

Distilled from RetinalFoundationHybridV2 (304M params) for deployment on
mid-range Android phones (4GB RAM, Tecno Spark 10).

Architecture:
    MobileNetV3-Large backbone (5.4M params, pretrained ImageNet)
    -> Feature projection (960 -> 512, LayerNorm + GELU)
    -> BottleneckClassifier (512 -> 128 -> num_classes)

The classifier head mirrors RetinalFoundationHybridV2's BottleneckClassifier
to preserve the precision-floor behaviour during distillation.

Target: INT8 ONNX <= 50 MB, p95 latency <= 1.8s on mid-range Android.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BottleneckClassifier(nn.Module):
    """Replicates the teacher's bottleneck head for structural compatibility.

    Architecture: input_dim -> hidden1 (drop1) -> hidden2 (drop2) -> num_classes
    Identical to retinal_foundation_hybrid_v2.BottleneckClassifier so feature-level
    distillation at the bottleneck layer is straightforward.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout1: float = 0.4,
        dropout2: float = 0.25,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MobileStudentV1(nn.Module):
    """MobileNetV3-Large student for on-device retinal screening.

    Design decisions:
    - MobileNetV3-Large chosen over EfficientNet-B0 for operator compatibility
      with the existing MobileNetV3-Small fundus gate model (shared ONNX op set).
    - Projection layer maps backbone features (960) to the teacher's hidden_dim
      (512) to enable feature-level distillation alignment.
    - BottleneckClassifier matches teacher structure for threshold compatibility.
    - Lower dropout than teacher (0.4/0.25 vs 0.5/0.3) since the student benefits
      from softer regularization during distillation.
    - No TTA: trained against teacher's TTA-averaged outputs but runs single-pass
      inference on device for latency compliance.

    Parameters
    ----------
    num_classes : int
        Number of disease classes (25-28 after ultra-rare pruning).
    hidden_dim : int
        Internal feature dimension for projection/classifier. Default 512.
    dropout1 : float
        Dropout after first bottleneck layer. Default 0.4.
    dropout2 : float
        Dropout after second bottleneck layer. Default 0.25.
    pretrained : bool
        Use ImageNet-pretrained MobileNetV3-Large backbone. Default True.
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_dim: int = 512,
        dropout1: float = 0.4,
        dropout2: float = 0.25,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # ---- MobileNetV3-Large backbone ----
        import timm

        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )
        # Probe actual output dim (timm's num_features can differ from
        # forward output due to final conv expansion in MobileNetV3)
        with torch.no_grad():
            _probe = self.backbone(torch.zeros(1, 3, 224, 224))
            backbone_dim = _probe.shape[-1]  # 1280 for MobileNetV3-Large

        # ---- Feature projection to teacher's hidden_dim ----
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Bottleneck Classifier (matches teacher structure) ----
        self.classifier = BottleneckClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        # ---- Per-class thresholds (loaded from teacher or re-optimized) ----
        self.register_buffer(
            "thresholds",
            torch.full((num_classes,), 0.5, dtype=torch.float32),
        )

        logger.info(
            "MobileStudentV1 initialized: backbone_dim=%d, hidden_dim=%d, "
            "num_classes=%d, params=%.2fM",
            backbone_dim,
            hidden_dim,
            num_classes,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits for loss computation."""
        features = self.backbone(x)             # [B, 960]
        projected = self.projection(features)   # [B, 512]
        logits = self.classifier(projected)     # [B, C]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract projected features (512-dim) for distillation alignment."""
        features = self.backbone(x)
        return self.projection(features)

    # ------------------------------------------------------------------
    # Threshold-aware inference (matches teacher's predict() interface)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference with per-class thresholds.

        Returns dict with:
          logits: raw logits [B, C]
          probabilities: sigmoid probabilities [B, C]
          predictions: binary predictions using thresholds [B, C]
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
        finally:
            if was_training:
                self.train()

        probs = torch.sigmoid(logits)
        preds = (probs >= self.thresholds.unsqueeze(0)).float()

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }

    def apply_thresholds(
        self, probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-class thresholds to probability tensor."""
        return (probabilities >= self.thresholds.unsqueeze(0)).float()

    # ------------------------------------------------------------------
    # Threshold I/O
    # ------------------------------------------------------------------

    def load_thresholds(self, path: str | Path) -> None:
        """Load per-class thresholds from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Format: {"disease_code": threshold_value, ...}
            vals = list(data.values())
        else:
            vals = data

        if len(vals) != self.num_classes:
            logger.warning(
                "Threshold count mismatch: file has %d, model has %d. "
                "Using first %d values.",
                len(vals),
                self.num_classes,
                min(len(vals), self.num_classes),
            )
            n = min(len(vals), self.num_classes)
            t = torch.full((self.num_classes,), 0.5)
            t[:n] = torch.tensor(vals[:n], dtype=torch.float32)
        else:
            t = torch.tensor(vals, dtype=torch.float32)

        self.thresholds.copy_(t)
        logger.info("Loaded thresholds from %s (mean=%.3f)", path, t.mean().item())

    def save_thresholds(self, path: str | Path) -> None:
        """Save per-class thresholds to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.thresholds.cpu().tolist(), f, indent=2)
        logger.info("Saved thresholds to %s", path)

    # ------------------------------------------------------------------
    # Export utilities
    # ------------------------------------------------------------------

    def prepare_for_export(self) -> "MobileStudentV1":
        """Prepare model for ONNX/TorchScript export.

        No-op for the student (no LoRA to merge), but maintains API
        compatibility with the teacher's export pipeline.
        """
        self.eval()
        logger.info("MobileStudentV1 prepared for export (no-op, no LoRA)")
        return self

    def get_param_summary(self) -> Dict[str, Any]:
        """Return parameter summary for model card generation."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "backbone": "mobilenetv3_large_100",
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "estimated_onnx_fp32_mb": total * 4 / 1e6,
            "estimated_onnx_int8_mb": total * 1 / 1e6,
        }

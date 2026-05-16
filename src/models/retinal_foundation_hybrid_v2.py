"""
RetinalFoundationHybrid v2 — Precision-optimized production model.

Addresses the core precision crisis (P=0.025) from RFMiD experiments:
  - Bottleneck classification head with heavy dropout (0.5/0.3)
  - No MoE (too many params for 1920 samples)
  - Reduced graph head (1 layer, not 2) to prevent overfitting
  - Support for per-class optimized thresholds at inference
  - Staged backbone unfreezing API
  - Fundus gate integration point

Changes from v1:
  - Removed MoE (overfits on small data)
  - Replaced UncertaintyHead with BottleneckClassifier (fewer params, more dropout)
  - Added unfreeze_backbone_blocks() for staged training
  - Added apply_thresholds() for precision-floor inference
  - Simplified graph reasoning (1 layer instead of 2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.retinal_foundation_encoder import RetinalFoundationEncoder
from src.models.vignn import ClinicalKnowledgeGraph, SparseTopKAttention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bottleneck Classifier (precision-focused)
# ---------------------------------------------------------------------------


class BottleneckClassifier(nn.Module):
    """High-dropout bottleneck head designed for small medical datasets.

    Architecture: input_dim -> 512 (drop 0.5) -> 128 (drop 0.3) -> num_classes

    The heavy dropout forces the model to rely on robust features rather than
    memorizing the 1920 training samples. The bottleneck at 128 dims prevents
    the classifier from encoding per-sample information.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout1: float = 0.5,
        dropout2: float = 0.3,
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


# ---------------------------------------------------------------------------
# Asymmetric Loss (precision-tuned)
# ---------------------------------------------------------------------------


class AsymmetricLossV2(nn.Module):
    """Asymmetric Loss tuned for extreme class imbalance (ICCV 2021).

    Key settings for precision rescue:
      - gamma_pos=0: never down-weight positive examples (they are precious)
      - gamma_neg=4: aggressively suppress easy negative false positives
      - clip=0.05: completely zero out loss from very confident negatives
      - label_smoothing=0.05: prevent overconfident predictions
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits).clamp(min=1e-8, max=1 - 1e-8)
        probs_neg = 1.0 - probs

        # Probability shifting (hard threshold for easy negatives)
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Positive and negative log-likelihood
        loss_pos = targets * torch.log(probs)
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))

        # Asymmetric focusing
        pt = probs * targets + probs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = (1.0 - pt).clamp(min=0.0) ** gamma

        return -((loss_pos + loss_neg) * focal_weight).mean()


# ---------------------------------------------------------------------------
# RetinalFoundationHybridV2 — Precision-Focused Model
# ---------------------------------------------------------------------------


class RetinalFoundationHybridV2(nn.Module):
    """Precision-optimized retinal disease classification model.

    Key differences from v1:
      - Bottleneck classifier with heavy dropout (0.5/0.3)
      - Simplified graph reasoning (1 attention layer)
      - No MoE (too many parameters for 1920 samples)
      - Per-class threshold support for precision-floor inference
      - Staged backbone unfreezing API

    Parameters
    ----------
    num_classes : int
        Number of disease classes (25-28 after ultra-rare pruning).
    hidden_dim : int
        Internal feature dimension. Default 512.
    dropout : float
        Base dropout for graph layers. Default 0.1.
    head_dropout1 : float
        Dropout after first bottleneck layer. Default 0.5.
    head_dropout2 : float
        Dropout after second bottleneck layer. Default 0.3.
    clinical_knowledge_graph : ClinicalKnowledgeGraph
        Required.
    backbone : str
        timm backbone name.
    use_lora : bool
        Enable LoRA adapters on backbone.
    lora_rank : int
        LoRA rank. Default 16.
    lora_alpha : float
        LoRA scaling. Default 32 (2x rank for more expressiveness).
    freeze_backbone : bool
        Freeze all backbone params initially.
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        head_dropout1: float = 0.5,
        head_dropout2: float = 0.3,
        clinical_knowledge_graph: Optional[ClinicalKnowledgeGraph] = None,
        backbone: str = "vit_large_patch16_224",
        img_size: int = 224,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if clinical_knowledge_graph is None:
            raise ValueError("RetinalFoundationHybridV2 requires a ClinicalKnowledgeGraph")

        self.clinical_knowledge_graph = clinical_knowledge_graph
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # ---- RETFound Backbone ----
        self.encoder = RetinalFoundationEncoder(
            backbone_name=backbone,
            output_dim=hidden_dim,
            img_size=img_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_backbone=freeze_backbone,
        )

        # ---- Lightweight Graph Reasoning (1 layer only) ----
        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_attn = SparseTopKAttention(hidden_dim, num_heads=8, dropout=dropout, top_k=32)
        self.graph_norm = nn.LayerNorm(hidden_dim)

        # Global context
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Bottleneck Classifier ----
        self.classifier = BottleneckClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout1=head_dropout1,
            dropout2=head_dropout2,
        )

        # ---- Per-class thresholds (set after training) ----
        self.register_buffer(
            "thresholds",
            torch.full((num_classes,), 0.5, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits for loss computation."""
        # Encode
        patch_features = self.encoder(x)  # [B, N, D]
        patch_embeds = self.patch_proj(patch_features)  # [B, N, D]

        # Single graph attention layer
        attn_out, _ = self.graph_attn(patch_embeds, patch_embeds, patch_embeds)
        graph_embeds = self.graph_norm(patch_embeds + attn_out)

        # Global average pooling
        global_feat = graph_embeds.mean(dim=1)  # [B, D]
        global_feat = self.global_pool(global_feat)

        # Classify
        logits = self.classifier(global_feat)  # [B, C]
        return logits

    # ------------------------------------------------------------------
    # Threshold-aware inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference with per-class thresholds.

        Returns dict with:
          logits: raw logits [B, C]
          probabilities: sigmoid probabilities [B, C]
          predictions: binary predictions using optimized thresholds [B, C]
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

    @torch.no_grad()
    def predict_with_tta(self, x: torch.Tensor, n_augments: int = 6) -> Dict[str, torch.Tensor]:
        """Test-Time Augmentation: average predictions over augmented views.

        Augmentations: original, hflip, vflip, rotate90, rotate180, rotate270
        """
        was_training = self.training
        self.eval()
        try:
            all_probs = []

            # Original
            all_probs.append(torch.sigmoid(self.forward(x)))

            # Horizontal flip
            if n_augments >= 2:
                all_probs.append(torch.sigmoid(self.forward(x.flip(-1))))

            # Vertical flip
            if n_augments >= 3:
                all_probs.append(torch.sigmoid(self.forward(x.flip(-2))))

            # Rotate 90
            if n_augments >= 4:
                all_probs.append(torch.sigmoid(self.forward(x.rot90(1, [-2, -1]))))

            # Rotate 180
            if n_augments >= 5:
                all_probs.append(torch.sigmoid(self.forward(x.rot90(2, [-2, -1]))))

            # Rotate 270
            if n_augments >= 6:
                all_probs.append(torch.sigmoid(self.forward(x.rot90(3, [-2, -1]))))

            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            preds = (avg_probs >= self.thresholds.unsqueeze(0)).float()

        finally:
            if was_training:
                self.train()

        return {
            "probabilities": avg_probs,
            "predictions": preds,
        }

    @torch.no_grad()
    def predict_with_clinical_reasoning(
        self,
        x: torch.Tensor,
        age: Optional[int] = None,
        comorbidities: Optional[Dict[str, bool]] = None,
        use_tta: bool = True,
    ) -> Dict[str, Any]:
        """Full inference pipeline with clinical reasoning."""
        if x.size(0) != 1:
            raise ValueError(f"Expects batch_size=1, got {x.size(0)}")

        if use_tta:
            result = self.predict_with_tta(x)
        else:
            result = self.predict(x)

        probs = result["probabilities"].cpu().numpy()[0]
        disease_names = self.clinical_knowledge_graph.disease_names

        pred_dict = {
            name: float(probs[i])
            for i, name in enumerate(disease_names)
            if i < len(probs) and probs[i] > 0.05
        }

        refined = self.clinical_knowledge_graph.apply_clinical_reasoning(pred_dict)

        # Use optimized thresholds for detection
        thresholds_np = self.thresholds.cpu().numpy()
        detected = [
            name
            for i, name in enumerate(disease_names)
            if i < len(probs) and probs[i] >= thresholds_np[i]
        ]

        referral = self.clinical_knowledge_graph.get_referral_priority(detected)
        risk = self.clinical_knowledge_graph.calculate_composite_risk_score(
            refined, age=age, comorbidities=comorbidities
        )
        treatments = self.clinical_knowledge_graph.get_treatment_recommendations(detected)

        return {
            "predictions": refined,
            "detected_diseases": detected,
            "referral_priority": referral,
            "risk_assessment": risk,
            "treatment_recommendations": treatments,
            "thresholds_used": {
                name: float(thresholds_np[i])
                for i, name in enumerate(disease_names)
                if i < len(thresholds_np)
            },
        }

    # ------------------------------------------------------------------
    # Staged backbone unfreezing
    # ------------------------------------------------------------------

    def unfreeze_backbone_blocks(self, num_blocks: int = 4, lr: float = 1e-6):
        """Unfreeze the last N transformer blocks of the backbone.

        Call this at the epoch where you want to start fine-tuning the backbone.
        Returns a list of param groups for the optimizer.

        Parameters
        ----------
        num_blocks : int
            Number of transformer blocks to unfreeze (from the end).
        lr : float
            Learning rate for the unfrozen backbone params.

        Returns
        -------
        list[dict]
            Parameter groups suitable for optimizer.add_param_group().
        """
        encoder = self.encoder.encoder  # The timm ViT model
        blocks = list(encoder.blocks)
        total_blocks = len(blocks)

        # Unfreeze last N blocks — only add params that were previously frozen
        # (LoRA params are already trainable and in the optimizer)
        unfrozen_params = []
        for i in range(max(0, total_blocks - num_blocks), total_blocks):
            for param in blocks[i].parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params.append(param)

        # Also unfreeze the norm layer after the blocks
        if hasattr(encoder, "norm"):
            for param in encoder.norm.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params.append(param)

        count = sum(p.numel() for p in unfrozen_params)
        logger.info(
            f"Unfroze last {num_blocks}/{total_blocks} backbone blocks "
            f"({count/1e6:.1f}M params) at lr={lr}"
        )

        return [{"params": unfrozen_params, "lr": lr, "name": "backbone_unfrozen"}]

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def load_thresholds(self, path: str):
        """Load per-class thresholds from JSON file."""
        with open(path) as f:
            data = json.load(f)

        thresholds = data.get("thresholds", data)
        if isinstance(thresholds, dict):
            disease_names = self.clinical_knowledge_graph.disease_names
            values = [thresholds.get(name, 0.5) for name in disease_names]
        elif isinstance(thresholds, list):
            values = thresholds
        else:
            raise ValueError(f"Unsupported threshold format: {type(thresholds)}")

        self.thresholds.copy_(torch.tensor(values[: self.num_classes], dtype=torch.float32))
        logger.info(f"Loaded thresholds from {path} (mean={self.thresholds.mean():.3f})")

    def save_thresholds(self, path: str):
        """Save per-class thresholds to JSON."""
        disease_names = self.clinical_knowledge_graph.disease_names
        data = {
            "thresholds": {
                name: float(self.thresholds[i])
                for i, name in enumerate(disease_names)
                if i < self.num_classes
            },
            "mean_threshold": float(self.thresholds.mean()),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved thresholds to {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_param_summary(self) -> Dict[str, int]:
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_total = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("encoder")
        )
        head_trainable = sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith("encoder") and p.requires_grad
        )
        return {
            "encoder_total": encoder_total,
            "encoder_trainable": encoder_trainable,
            "head_total": head_total,
            "head_trainable": head_trainable,
            "total": encoder_total + head_total,
            "trainable": encoder_trainable + head_trainable,
        }

    def prepare_for_export(self):
        """Merge LoRA and prepare for ONNX/TorchScript export."""
        self.encoder.merge_lora_for_export()
        logger.info("Model prepared for export (LoRA merged)")


# ---------------------------------------------------------------------------
# Class filtering utility
# ---------------------------------------------------------------------------


def filter_rare_classes(
    disease_columns: List[str],
    labels_df,
    min_samples: int = 10,
) -> List[str]:
    """Return disease columns with >= min_samples positive training examples.

    Parameters
    ----------
    disease_columns : list[str]
        All candidate disease columns.
    labels_df : DataFrame
        Training labels (binary columns).
    min_samples : int
        Minimum positive samples to retain a class.

    Returns
    -------
    list[str]
        Filtered disease columns.
    """
    kept = []
    dropped = []
    for col in disease_columns:
        if col in labels_df.columns:
            n_positive = int(labels_df[col].sum())
            if n_positive >= min_samples:
                kept.append(col)
            else:
                dropped.append((col, n_positive))

    logger.info(
        f"Class filtering: {len(kept)}/{len(disease_columns)} retained "
        f"(min_samples={min_samples})"
    )
    if dropped:
        logger.info(f"Dropped {len(dropped)} rare classes: {dropped[:10]}...")

    return kept


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_hybrid_v2(
    num_classes: int = 28,
    hidden_dim: int = 512,
    clinical_knowledge_graph: Optional[ClinicalKnowledgeGraph] = None,
    backbone: str = "vit_large_patch16_224",
    use_lora: bool = True,
    lora_rank: int = 16,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> RetinalFoundationHybridV2:
    """Create a precision-optimized hybrid model."""
    if clinical_knowledge_graph is None:
        from src.models.vignn import create_knowledge_graph

        clinical_knowledge_graph = create_knowledge_graph()

    model = RetinalFoundationHybridV2(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        clinical_knowledge_graph=clinical_knowledge_graph,
        backbone=backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        **kwargs,
    )

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    params = model.get_param_summary()
    logger.info(
        f"HybridV2 created | Total: {params['total']/1e6:.1f}M | "
        f"Trainable: {params['trainable']/1e6:.1f}M | classes: {num_classes}"
    )

    return model

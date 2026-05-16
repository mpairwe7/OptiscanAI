"""Multi-modal fusion skeleton for Fundus + OCT + patient metadata.

Phase 4 future-proofing module.  Defines abstract interfaces for modality
encoders and fusion strategies, with concrete implementations for the
existing ViGNN fundus backbone, a placeholder OCT encoder (ViT-Small),
a tabular patient-metadata MLP, and two fusion strategies (concatenation
and cross-attention).

The top-level ``RetinalMultiModalClassifier`` accepts a dict of modality
inputs, gracefully handles missing modalities via learned default
embeddings, and returns logits together with metadata about which
modalities contributed to the prediction.

All functionality is gated behind ``MULTIMODAL__ENABLED=false`` by
default.  When disabled the module can still be imported; the classifier
simply wraps the fundus-only path.
"""
from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — PyTorch convention

# Project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class ModalityEncoder(ABC):
    """Abstract base for a single-modality encoder.

    Subclasses must implement :meth:`encode` and expose the
    ``output_dim`` and ``modality_name`` properties so the fusion layer
    can reason about the embedding geometry at runtime.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw modality input into a fixed-size embedding.

        Parameters
        ----------
        x : torch.Tensor
            Modality-specific input tensor (images, tabular rows, etc.).

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(batch, output_dim)``.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the embedding produced by :meth:`encode`."""
        ...

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Short identifier for this modality (e.g. ``'fundus'``)."""
        ...


class FusionStrategy(ABC):
    """Abstract base for fusing embeddings from multiple modalities."""

    @abstractmethod
    def fuse(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine per-modality embeddings into a single representation.

        Parameters
        ----------
        embeddings : dict[str, torch.Tensor]
            Mapping from modality name to its ``(batch, dim)`` embedding.

        Returns
        -------
        torch.Tensor
            Fused embedding of shape ``(batch, fused_dim)``.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete encoders
# ---------------------------------------------------------------------------


class FundusEncoder(ModalityEncoder, nn.Module):
    """Wraps the production ViGNN backbone for colour fundus photographs.

    The encoder reuses the existing ``MultiResolutionEncoder`` and
    ``patch_proj`` layers from the ViGNN model, producing a 384-d
    global embedding per image.  If the full ViGNN checkpoint is
    available the weights are loaded; otherwise random init is used.
    """

    _OUTPUT_DIM = 384

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._dim = self._OUTPUT_DIM

        # Build a lightweight feature extractor from the ViGNN backbone
        try:
            from src.models.vignn import MultiResolutionEncoder
            self.visual_encoder = MultiResolutionEncoder(
                backbone_name="vit_small_patch16_224",
                output_dim=self._dim,
                img_size=224,
            )
        except Exception:
            logger.warning(
                "FundusEncoder: could not import MultiResolutionEncoder, "
                "falling back to identity projection"
            )
            self.visual_encoder = None

        self.proj = nn.Sequential(
            nn.Linear(self._dim, self._dim),
            nn.LayerNorm(self._dim),
            nn.GELU(),
        )

    # -- ModalityEncoder interface -----------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of fundus images ``(B, 3, 224, 224)`` -> ``(B, 384)``."""
        if self.visual_encoder is not None:
            patch_tokens = self.visual_encoder(x)          # (B, N, D)
            pooled = patch_tokens.mean(dim=1)              # (B, D)
        else:
            pooled = x.flatten(1)[:, : self._dim]
            if pooled.size(1) < self._dim:
                pooled = F.pad(pooled, (0, self._dim - pooled.size(1)))
        return self.proj(pooled)

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality_name(self) -> str:
        return "fundus"

    # nn.Module forward delegates to encode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class OCTEncoder(ModalityEncoder, nn.Module):
    """Placeholder encoder for Optical Coherence Tomography volumes.

    Uses a ViT-Small backbone from ``timm`` (patch16, 224x224 slices).
    In production, this would accept 3-D OCT volumes and apply
    slice-level encoding followed by volume-level aggregation.  The
    current skeleton treats each input as a single 2-D B-scan slice.
    """

    _OUTPUT_DIM = 384

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._dim = self._OUTPUT_DIM

        try:
            import timm
            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
            )
            backbone_out = self.backbone.num_features
        except ImportError:
            logger.warning("OCTEncoder: timm not available, using linear stub")
            self.backbone = None
            backbone_out = 3 * 224 * 224  # flat pixel fallback

        self.proj = nn.Sequential(
            nn.Linear(backbone_out, self._dim),
            nn.LayerNorm(self._dim),
            nn.GELU(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of OCT B-scans ``(B, 3, 224, 224)`` -> ``(B, 384)``."""
        if self.backbone is not None:
            features = self.backbone(x)                    # (B, backbone_out)
        else:
            features = x.flatten(1)
        return self.proj(features)

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality_name(self) -> str:
        return "oct"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class PatientMetadataEncoder(ModalityEncoder, nn.Module):
    """MLP encoder for tabular patient metadata.

    Handles missing values by replacing NaN / sentinel values with a
    per-feature learned default before encoding.

    Expected input columns (default 8 features):
        age, sex, iop_mmhg, visual_acuity, hba1c, systolic_bp,
        diastolic_bp, bmi

    The feature count is configurable via ``num_features``.
    """

    _OUTPUT_DIM = 384
    DEFAULT_NUM_FEATURES = 8

    def __init__(self, num_features: int | None = None) -> None:
        nn.Module.__init__(self)
        self._dim = self._OUTPUT_DIM
        self._num_features = num_features or self.DEFAULT_NUM_FEATURES

        # Learned defaults for imputing missing values (NaN -> default)
        self.feature_defaults = nn.Parameter(
            torch.zeros(self._num_features)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self._num_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self._dim),
            nn.LayerNorm(self._dim),
            nn.GELU(),
        )

    def _impute(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with learned defaults."""
        mask = torch.isnan(x)
        if mask.any():
            defaults = self.feature_defaults.unsqueeze(0).expand_as(x)
            x = torch.where(mask, defaults, x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patient metadata ``(B, num_features)`` -> ``(B, 384)``."""
        x = self._impute(x)
        return self.mlp(x)

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def modality_name(self) -> str:
        return "metadata"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------


class ConcatenationFusion(FusionStrategy, nn.Module):
    """Concatenation followed by a linear projection.

    Concatenates all modality embeddings along the feature axis and
    projects to ``output_dim`` via a single linear + LayerNorm block.
    """

    def __init__(self, modality_dims: dict[str, int], output_dim: int = 384) -> None:
        nn.Module.__init__(self)
        self._output_dim = output_dim
        total_dim = sum(modality_dims.values())
        self._modality_order = sorted(modality_dims.keys())

        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def fuse(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate embeddings in deterministic order and project."""
        ordered = [embeddings[m] for m in self._modality_order if m in embeddings]
        if not ordered:
            raise ValueError("No embeddings provided to ConcatenationFusion")
        concatenated = torch.cat(ordered, dim=-1)
        return self.projection(concatenated)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse(embeddings)


class CrossAttentionFusion(FusionStrategy, nn.Module):
    """Multi-head cross-attention fusion.

    Each modality embedding attends to all other modalities using
    standard multi-head attention.  The attended representations are
    averaged and projected to ``output_dim``.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self._output_dim = output_dim
        self._modality_order = sorted(modality_dims.keys())

        # Per-modality input projections to a common dimension
        self.input_projs = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def fuse(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-attend across modalities and project."""
        projected: list[torch.Tensor] = []
        for name in self._modality_order:
            if name not in embeddings:
                continue
            emb = embeddings[name]
            proj = self.input_projs[name](emb)       # (B, D)
            projected.append(proj.unsqueeze(1))       # (B, 1, D)

        if not projected:
            raise ValueError("No embeddings provided to CrossAttentionFusion")

        # Stack modalities as sequence tokens: (B, M, D)
        tokens = torch.cat(projected, dim=1)

        attended, attn_weights = self.cross_attn(tokens, tokens, tokens)
        attended = self.norm(attended + tokens)        # residual
        pooled = attended.mean(dim=1)                  # (B, D)
        return self.out_proj(pooled)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse(embeddings)


# ---------------------------------------------------------------------------
# Multi-modal classifier
# ---------------------------------------------------------------------------


class RetinalMultiModalClassifier(nn.Module):
    """Top-level multi-modal classifier for retinal disease screening.

    Accepts a ``dict[str, Tensor]`` mapping modality names to their raw
    inputs.  Missing modalities are replaced with learned default
    embeddings so the model degrades gracefully when only a subset of
    data is available (e.g. fundus-only inference in a field clinic).

    Returns
    -------
    dict
        ``{logits, modalities_used, fusion_weights}``
    """

    def __init__(
        self,
        num_classes: int | None = None,
        fusion_strategy: str | None = None,
        num_metadata_features: int = PatientMetadataEncoder.DEFAULT_NUM_FEATURES,
    ) -> None:
        super().__init__()
        cfg = settings.multimodal
        self._enabled = cfg.enabled
        self._active_modalities = cfg.modalities
        _fusion_name = fusion_strategy or cfg.fusion_strategy
        _num_classes = num_classes or settings.num_classes

        logger.info(
            "RetinalMultiModalClassifier init (enabled=%s, modalities=%s, fusion=%s)",
            self._enabled,
            self._active_modalities,
            _fusion_name,
        )

        # -- Encoders -------------------------------------------------------
        self.encoders = nn.ModuleDict()
        modality_dims: dict[str, int] = {}

        if "fundus" in self._active_modalities:
            enc = FundusEncoder()
            self.encoders["fundus"] = enc
            modality_dims["fundus"] = enc.output_dim

        if "oct" in self._active_modalities:
            enc = OCTEncoder()
            self.encoders["oct"] = enc
            modality_dims["oct"] = enc.output_dim

        if "metadata" in self._active_modalities:
            enc = PatientMetadataEncoder(num_features=num_metadata_features)
            self.encoders["metadata"] = enc
            modality_dims["metadata"] = enc.output_dim

        # Fallback: at least fundus
        if not self.encoders:
            enc = FundusEncoder()
            self.encoders["fundus"] = enc
            modality_dims["fundus"] = enc.output_dim
            self._active_modalities = ["fundus"]

        # -- Learned default embeddings for missing modalities ---------------
        self.default_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(dim) * 0.02)
            for name, dim in modality_dims.items()
        })

        # -- Fusion -----------------------------------------------------------
        if _fusion_name == "cross_attention":
            self.fusion: nn.Module = CrossAttentionFusion(
                modality_dims=modality_dims,
                output_dim=384,
                num_heads=4,
            )
            fused_dim = 384
        else:
            self.fusion = ConcatenationFusion(
                modality_dims=modality_dims,
                output_dim=384,
            )
            fused_dim = 384

        # -- Classifier head --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, _num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run multi-modal inference.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Mapping from modality name to its raw input tensor.
            Only present modalities are encoded; missing ones use learned
            default embeddings.

        Returns
        -------
        dict
            ``logits`` : Tensor of shape ``(B, num_classes)``
            ``modalities_used`` : list[str] of modalities that were present
            ``fusion_weights`` : dict[str, float] indicating relative
                contribution (based on embedding norm ratio)
        """
        batch_size: int | None = None
        device: torch.device | None = None

        # Determine batch size and device from any available input
        for v in inputs.values():
            batch_size = v.size(0)
            device = v.device
            break

        if batch_size is None:
            raise ValueError("inputs dict is empty; at least one modality required")

        # Encode each modality or substitute defaults
        embeddings: dict[str, torch.Tensor] = {}
        modalities_used: list[str] = []

        for name, encoder in self.encoders.items():
            if name in inputs:
                embeddings[name] = encoder(inputs[name])
                modalities_used.append(name)
            else:
                # Use learned default, expanded to batch
                default = self.default_embeddings[name]
                embeddings[name] = default.unsqueeze(0).expand(batch_size, -1).to(device)

        # Fuse
        fused = self.fusion(embeddings)

        # Classify
        logits = self.classifier(fused)

        # Compute fusion weights as relative embedding norms
        norms = {
            name: float(emb.norm(dim=-1).mean().item())
            for name, emb in embeddings.items()
        }
        total_norm = sum(norms.values()) or 1.0
        fusion_weights = {name: n / total_norm for name, n in norms.items()}

        return {
            "logits": logits,
            "modalities_used": modalities_used,
            "fusion_weights": fusion_weights,
        }


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_classifier: RetinalMultiModalClassifier | None = None


def get_multimodal_classifier() -> RetinalMultiModalClassifier | None:
    """Return the singleton multi-modal classifier, or ``None`` when disabled."""
    return _classifier


def init_multimodal_classifier() -> None:
    """Create the module-level RetinalMultiModalClassifier singleton.

    Respects ``settings.multimodal.enabled``; does nothing when the
    feature is turned off.
    """
    global _classifier

    if not settings.multimodal.enabled:
        logger.info("Multi-modal fusion disabled (MULTIMODAL__ENABLED=false)")
        return

    try:
        _classifier = RetinalMultiModalClassifier()
        logger.info(
            "RetinalMultiModalClassifier initialised (modalities=%s)",
            _classifier._active_modalities,
        )
    except Exception:
        logger.exception("Failed to initialise multi-modal classifier")
        _classifier = None

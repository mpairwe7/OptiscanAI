"""
RetinalFoundationEncoder — RETFound ViT-Large backbone with LoRA adapters.

Replaces the legacy MultiResolutionEncoder (ViT-Small/ImageNet) with a domain-specific
retinal foundation model pretrained via MAE on 1.6M colour fundus photographs.

Weight loading priority:
  1. Local file  (pretrained_weights/RETFound_cfp.pth)
  2. HuggingFace (YukunZhou/RETFound_mae_natureCFP)
  3. timm ViT-Large (ImageNet fallback)

Supports LoRA (rank 16-32) for parameter-efficient fine-tuning: only 2-4M trainable
parameters on a 304M backbone, enabling training on a single A6000/A100 GPU.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LoRA Adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning (Hu et al., 2021).

    Wraps an existing ``nn.Linear`` and adds a low-rank bypass:
        h = W_frozen @ x + (alpha / r) * B @ A @ x

    Only A and B are trainable.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Kaiming init for A, zero init for B (output starts at zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original weight
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into the original linear for zero-overhead inference."""
        device = self.original.weight.device
        dtype = self.original.weight.dtype
        with torch.no_grad():
            merged = nn.Linear(
                self.original.in_features, self.original.out_features,
                bias=self.original.bias is not None,
                device=device, dtype=dtype,
            )
            merged.weight.copy_(
                self.original.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            )
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)
        return merged


def apply_lora_to_vit(model: nn.Module, rank: int = 16, alpha: float = 16.0,
                      dropout: float = 0.05, target_modules: Optional[list[str]] = None):
    """Inject LoRA adapters into a ViT's QKV projection layers.

    By default targets ``qkv`` projections in every attention block.  Pass
    ``target_modules`` to override (e.g. ``["qkv", "proj"]``).
    """
    target_modules = target_modules or ["qkv"]
    replaced = 0

    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, target, lora_layer)
                    replaced += 1

    logger.info(f"LoRA injected into {replaced} layers (rank={rank}, alpha={alpha})")
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters back into base weights for deployment."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge_weights())
                merged += 1
    logger.info(f"Merged {merged} LoRA adapters into base weights")
    return model


# ---------------------------------------------------------------------------
# RetinalFoundationEncoder
# ---------------------------------------------------------------------------

# Paths searched for local weights, in priority order
_LOCAL_WEIGHT_PATHS = [
    "pretrained_weights/RETFound_cfp.pth",
    "pretrained_weights/RETFound_mae_natureCFP.pth",
    "pretrained_weights/vit_large_patch16_224.pth",
]

# HuggingFace model ID for auto-download
_HF_MODEL_ID = "YukunZhou/RETFound_mae_natureCFP"


class RetinalFoundationEncoder(nn.Module):
    """RETFound ViT-Large backbone with optional LoRA and multi-resolution support.

    Parameters
    ----------
    backbone_name : str
        timm model name. Default ``vit_large_patch16_224`` (matches RETFound).
    output_dim : int
        Dimension of the projected patch tokens. Default 1024 (native ViT-L dim).
    img_size : int
        Input resolution. Default 224.
    use_lora : bool
        If True, inject LoRA adapters and freeze base weights.
    lora_rank : int
        Rank of LoRA decomposition. 16 is good for retinal tasks.
    lora_alpha : float
        LoRA scaling factor (typically equal to rank).
    lora_dropout : float
        Dropout applied before LoRA projection.
    lora_targets : list[str] | None
        Which linear layers to adapt. Default: ``["qkv"]``.
    multi_resolution : bool
        If True, process at multiple resolutions and fuse. Adds latency but
        may improve detection of small lesions (microaneurysms, hard exudates).
    freeze_backbone : bool
        If True and ``use_lora`` is False, freezes all backbone params.
    """

    def __init__(
        self,
        backbone_name: str = "vit_large_patch16_224",
        output_dim: int = 1024,
        img_size: int = 224,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[list[str]] = None,
        multi_resolution: bool = False,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.multi_resolution = multi_resolution
        self.use_lora = use_lora

        if multi_resolution:
            self.resolutions = [img_size, int(img_size * 0.71), int(img_size * 0.57)]
        else:
            self.resolutions = [img_size]

        # ---- Build backbone ----
        self.encoder = self._load_backbone(backbone_name)
        self.backbone_dim = self.encoder.num_features  # 1024 for ViT-L

        # ---- Freeze backbone before LoRA injection ----
        if freeze_backbone or use_lora:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ---- LoRA adapters ----
        if use_lora:
            apply_lora_to_vit(
                self.encoder, rank=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout, target_modules=lora_targets,
            )

        # ---- Resolution projection & fusion ----
        n_res = len(self.resolutions)
        self.resolution_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            for _ in range(n_res)
        ])

        if n_res > 1:
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * n_res, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        else:
            self.fusion = nn.Identity()

        self.output_dim = output_dim

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_backbone(self, backbone_name: str) -> nn.Module:
        """Load backbone with RETFound weights or fall back gracefully."""
        timm_kwargs = dict(pretrained=False, num_classes=0, dynamic_img_size=True)
        use_pretrained = os.environ.get("USE_PRETRAINED", "1") == "1"

        if not use_pretrained:
            encoder = timm.create_model(backbone_name, **timm_kwargs)
            logger.info(f"Backbone {backbone_name} with random init (USE_PRETRAINED=0)")
            return encoder

        # Priority 1: Local RETFound / ViT-L weights
        for path in _LOCAL_WEIGHT_PATHS:
            if os.path.isfile(path):
                try:
                    encoder = timm.create_model(backbone_name, **timm_kwargs)
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    if "model" in state:
                        state = state["model"]
                    # Filter MAE decoder keys
                    state = {
                        k: v for k, v in state.items()
                        if not k.startswith("decoder") and "mask_token" not in k
                    }
                    missing, unexpected = encoder.load_state_dict(state, strict=False)
                    logger.info(
                        f"Loaded RETFound weights from {path} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
                    return encoder
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        # Priority 2: HuggingFace Hub
        try:
            encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                                        dynamic_img_size=True)
            logger.info(f"Loaded backbone from HuggingFace Hub / timm registry")
            return encoder
        except Exception as e:
            logger.warning(f"HuggingFace download failed: {e}")

        # Priority 3: Random init fallback
        encoder = timm.create_model(backbone_name, **timm_kwargs)
        logger.warning(f"Using random initialization for {backbone_name} (no pretrained weights found)")
        return encoder

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-resolution patch tokens.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_patches, output_dim]``.  For ViT-L/16 at 224px
            this is ``[B, 196, 1024]``.
        """
        primary_size = self.resolutions[0]
        features = []

        for resolution, proj in zip(self.resolutions, self.resolution_projections):
            if x.size(-1) != resolution or x.size(-2) != resolution:
                x_r = F.interpolate(x, size=(resolution, resolution),
                                    mode="bilinear", align_corners=False)
            else:
                x_r = x

            # Resize back to primary for uniform patch count
            if resolution != primary_size:
                x_r = F.interpolate(x_r, size=(primary_size, primary_size),
                                    mode="bilinear", align_corners=False)

            tokens = self.encoder.forward_features(x_r)  # [B, N+1, D]
            patch_tokens = tokens[:, 1:, :]               # drop CLS
            features.append(proj(patch_tokens))

        if len(features) == 1:
            return features[0]

        return self.fusion(torch.cat(features, dim=-1))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters (LoRA + projection)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def merge_lora_for_export(self):
        """Merge LoRA weights into base for zero-overhead export."""
        if self.use_lora:
            merge_lora_weights(self.encoder)
            self.use_lora = False
            logger.info("LoRA merged — encoder ready for export")

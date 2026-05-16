"""Fundus Gate V2 — production-hardened fusion of statistical and learned gates.

Combines the fast rule-based statistical gate with a MobileNetV3-Small binary
classifier for robust pre-inference rejection of non-fundus images.

Architecture:
    1. Statistical gate (fast path, <5ms) — immediate reject on failure
    2. Learned gate (MobileNetV3-Small, <6ms) — runs only if statistical passes
    3. Fusion: confidence = (1 - learned_weight) * stat + learned_weight * learned
    4. Hard spatial requirement: (dark_border OR circular_aperture) AND radial_sharpness

Safety:
    - Fallback to statistical-only if learned model unavailable
    - Thread-safe (no mutable state after init)
    - Configurable via environment variables
    - Explainable rejection with structured diagnostics and visual evidence

Usage:
    from src.data.fundus_gate_v2 import gate_image, gate_predictions

    result = gate_image(pil_image)
    if not result.passed:
        # result.reason, result.failed_checks, result.visual_evidence
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from src.data.fundus_gate import (
    GateResult,
    check_statistical,
    check_structural,
    gate_predictions,  # noqa: F401 — re-exported
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ENV_PREFIX = "FUNDUS_GATE__"


def _env(key: str, default: str) -> str:
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_bool(key: str, default: bool) -> bool:
    val = _env(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# GateResultV2
# ---------------------------------------------------------------------------


@dataclass
class GateResultV2(GateResult):
    """Extended gate result with fusion decomposition and visual evidence."""

    statistical_confidence: float = 0.0
    learned_confidence: Optional[float] = None  # None = learned gate unavailable
    fused_confidence: float = 0.0
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"statistical": 0.6, "learned": 0.4}
    )
    visual_evidence: Optional[dict[str, str]] = None  # key -> base64 data URI
    suggested_action: str = ""
    gate_version: str = "v2"
    latency_ms: float = 0.0
    failed_checks: Optional[list[dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# FundusGateV2
# ---------------------------------------------------------------------------


class FundusGateV2:
    """Production-hardened fusion gate combining statistical and learned approaches.

    Thread-safe: no mutable state after __init__. The learned model is in
    eval() mode with torch.no_grad(). PIL operations create fresh arrays.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        learned_weight: float | None = None,
        min_confidence: float | None = None,
        model_path: str | None = None,
        visual_evidence: bool | None = None,
        mc_dropout_samples: int | None = None,
    ):
        self.enabled = enabled if enabled is not None else _env_bool("ENABLED", True)
        self.learned_weight = (
            learned_weight if learned_weight is not None else _env_float("LEARNED_WEIGHT", 0.4)
        )
        self.min_confidence = (
            min_confidence if min_confidence is not None else _env_float("MIN_CONFIDENCE", 0.70)
        )
        self._model_path = model_path or _env("MODEL_PATH", "weights/fundus_gate.pth")
        self._visual_evidence = (
            visual_evidence if visual_evidence is not None else _env_bool("VISUAL_EVIDENCE", False)
        )
        self._mc_samples = (
            mc_dropout_samples if mc_dropout_samples is not None else _env_int("MC_DROPOUT_SAMPLES", 5)
        )

        # Attempt to load learned gate — fallback to statistical-only on failure
        self._learned_gate = None
        self._learned_available = False
        self._gradcam_target_layer = None
        self._load_learned_gate()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_learned_gate(self) -> None:
        """Load MobileNetV3-Small learned gate. Non-fatal on failure."""
        try:
            from src.data.fundus_gate_learned import LearnedFundusGate

            path = self._model_path if os.path.isfile(self._model_path) else None
            self._learned_gate = LearnedFundusGate(weights_path=path, threshold=0.5)
            self._learned_gate.eval()
            self._learned_available = True

            # Identify GradCAM target layer for visual evidence
            self._gradcam_target_layer = self._find_gradcam_layer()

            if path:
                logger.info("Fundus gate v2: learned model loaded from %s", path)
            else:
                logger.warning(
                    "Fundus gate v2: learned model initialized with ImageNet weights "
                    "(no trained weights at %s)",
                    self._model_path,
                )
        except Exception as e:
            logger.warning(
                "Fundus gate v2: learned gate unavailable (%s). "
                "Falling back to statistical-only mode.",
                e,
            )
            self._learned_gate = None
            self._learned_available = False

    def _find_gradcam_layer(self):
        """Find the last convolutional block for GradCAM activation maps."""
        if self._learned_gate is None:
            return None
        try:
            backbone = self._learned_gate.backbone
            # timm MobileNetV3: backbone.blocks is a Sequential of InvertedResidual
            if hasattr(backbone, "blocks"):
                return backbone.blocks[-1]
            # Fallback CNN: find last Conv2d
            last_conv = None
            for module in backbone.modules():
                import torch.nn as nn
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            return last_conv
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core gate logic
    # ------------------------------------------------------------------

    def gate_image(self, image: Image.Image) -> GateResultV2:
        """Run the full v2 fusion gate pipeline.

        Flow:
            1. If disabled -> immediate pass
            2. Structural checks -> fast reject
            3. Statistical checks -> fast reject if fails
            4. Learned gate (if available) -> run on statistical pass
            5. Fusion -> weighted combination
            6. Hard spatial requirement enforcement
            7. Explainable rejection on failure
        """
        t0 = time.perf_counter()

        # Gate disabled — pass everything
        if not self.enabled:
            return GateResultV2(
                passed=True,
                confidence=1.0,
                reason="Gate disabled",
                checks={},
                layer="none",
                fused_confidence=1.0,
                gate_version="v2",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # --- Layer 1: Structural ---
        struct_passed, struct_checks = check_structural(image)
        if not struct_passed:
            failed = [k for k, v in struct_checks.items() if not v.get("passed", True)]
            reason, action = self._build_structural_message(struct_checks, failed)
            return GateResultV2(
                passed=False,
                confidence=0.0,
                reason=reason,
                checks={"structural": struct_checks},
                layer="structural",
                statistical_confidence=0.0,
                learned_confidence=None,
                fused_confidence=0.0,
                fusion_weights={"statistical": 1.0, "learned": 0.0},
                suggested_action=action,
                gate_version="v2",
                latency_ms=(time.perf_counter() - t0) * 1000,
                failed_checks=[
                    {"name": name, "value": struct_checks[name], "threshold": "pass"}
                    for name in failed
                ],
            )

        # --- Layer 2: Statistical ---
        stat_passed, stat_confidence, stat_checks = check_statistical(image)

        # Extract hard spatial requirement
        has_fundus_spatial = (
            (stat_checks.get("dark_border", {}).get("passed", False)
             or stat_checks.get("circular_aperture", {}).get("passed", False))
            and stat_checks.get("radial_sharpness", {}).get("passed", False)
            and stat_checks.get("hue_concentration", {}).get("passed", False)
        )

        # If statistical fails, reject immediately — no learned gate needed
        if not stat_passed:
            reason, action = self._build_statistical_message(stat_confidence, stat_checks)
            evidence = None
            if self._visual_evidence:
                evidence = self._generate_evidence(image, stat_checks, None)
            return GateResultV2(
                passed=False,
                confidence=stat_confidence,
                reason=reason,
                checks={"structural": struct_checks, "statistical": stat_checks},
                layer="statistical",
                statistical_confidence=stat_confidence,
                learned_confidence=None,
                fused_confidence=stat_confidence,
                fusion_weights={"statistical": 1.0, "learned": 0.0},
                visual_evidence=evidence,
                suggested_action=action,
                gate_version="v2",
                latency_ms=(time.perf_counter() - t0) * 1000,
                failed_checks=self._build_failed_checks_list(stat_checks, None, stat_confidence),
            )

        # --- Layer 3: Learned gate ---
        learned_prob = None
        learned_msg = ""
        if self._learned_available and self._learned_gate is not None:
            try:
                _, learned_prob, learned_msg = self._learned_gate.check(image)
            except Exception as e:
                logger.warning("Learned gate inference failed: %s", e)
                learned_prob = None

        # --- Fusion ---
        if learned_prob is not None:
            stat_weight = 1.0 - self.learned_weight
            fused = stat_weight * stat_confidence + self.learned_weight * learned_prob
            weights = {"statistical": round(stat_weight, 2), "learned": round(self.learned_weight, 2)}
        else:
            # Fallback: statistical-only
            fused = stat_confidence
            weights = {"statistical": 1.0, "learned": 0.0}

        # --- Decision ---
        passed = fused >= self.min_confidence and has_fundus_spatial

        if passed:
            return GateResultV2(
                passed=True,
                confidence=fused,
                reason="Image passes fundus validation (v2 fusion gate)",
                checks={"structural": struct_checks, "statistical": stat_checks},
                layer="fusion",
                statistical_confidence=stat_confidence,
                learned_confidence=learned_prob,
                fused_confidence=fused,
                fusion_weights=weights,
                suggested_action="",
                gate_version="v2",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # --- Rejection ---
        reason, action = self._build_fusion_message(
            stat_confidence, learned_prob, fused, has_fundus_spatial, stat_checks
        )
        if not self._learned_available and learned_prob is None:
            reason += " (note: learned gate unavailable; statistical-only mode)"

        evidence = None
        if self._visual_evidence:
            evidence = self._generate_evidence(image, stat_checks, learned_prob)

        layer = "fusion"
        if not has_fundus_spatial:
            layer = "fusion"  # spatial requirement is part of fusion decision

        return GateResultV2(
            passed=False,
            confidence=fused,
            reason=reason,
            checks={"structural": struct_checks, "statistical": stat_checks},
            layer=layer,
            statistical_confidence=stat_confidence,
            learned_confidence=learned_prob,
            fused_confidence=fused,
            fusion_weights=weights,
            visual_evidence=evidence,
            suggested_action=action,
            gate_version="v2",
            latency_ms=(time.perf_counter() - t0) * 1000,
            failed_checks=self._build_failed_checks_list(stat_checks, learned_prob, fused),
        )

    # ------------------------------------------------------------------
    # Rejection message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_structural_message(
        checks: dict, failed: list[str]
    ) -> tuple[str, str]:
        parts = []
        action = "Please upload a color retinal fundus photograph."
        for name in failed:
            info = checks.get(name, {})
            if name == "resolution":
                w, h = info.get("width", 0), info.get("height", 0)
                parts.append(f"resolution too low ({w}x{h}px, minimum 100x100)")
                action = (
                    "Please upload a higher resolution retinal fundus photograph "
                    "(minimum 100x100 pixels)."
                )
            elif name == "aspect_ratio":
                parts.append(
                    f"unusual aspect ratio ({info.get('ratio', '?')}:1, "
                    f"deviation {info.get('deviation', '?')})"
                )
                action = (
                    "Please upload a retinal fundus photograph with standard aspect ratio "
                    "(approximately 1:1 or 4:3)."
                )
            elif name == "color_mode":
                parts.append(f"unsupported color mode ({info.get('mode', '?')}, requires RGB)")
                action = "Please upload a color (RGB) retinal fundus photograph."
        reason = f"Image failed structural checks: {'; '.join(parts)}."
        return reason, action

    @staticmethod
    def _build_statistical_message(
        confidence: float, checks: dict
    ) -> tuple[str, str]:
        reason = (
            f"Image does not match retinal fundus color and spatial profile. "
            f"Fundus confidence: {confidence:.0%}."
        )
        action = (
            "Please upload a color retinal fundus photograph taken with a "
            "dedicated fundus camera or validated smartphone adapter."
        )
        return reason, action

    @staticmethod
    def _build_fusion_message(
        stat_conf: float,
        learned_conf: float | None,
        fused: float,
        has_spatial: bool,
        stat_checks: dict,
    ) -> tuple[str, str]:
        parts = [f"statistical: {stat_conf:.0%}"]
        if learned_conf is not None:
            parts.append(f"learned: {learned_conf:.0%}")
        parts.append(f"fused: {fused:.0%}")

        spatial_note = ""
        if not has_spatial:
            spatial_note = " Failed radial boundary sharpness check."

        reason = (
            f"Image partially matches retinal fundus characteristics "
            f"({', '.join(parts)}) but falls below the clinical confidence "
            f"threshold.{spatial_note}"
        )
        action = (
            "Please upload a color retinal fundus photograph taken with a "
            "dedicated fundus camera or validated smartphone adapter."
        )
        return reason, action

    # ------------------------------------------------------------------
    # Structured failed checks
    # ------------------------------------------------------------------

    @staticmethod
    def _build_failed_checks_list(
        stat_checks: dict,
        learned_conf: float | None,
        fused_conf: float,
    ) -> list[dict[str, Any]]:
        failed = []
        for name, info in stat_checks.items():
            if isinstance(info, dict) and not info.get("passed", True):
                entry: dict[str, Any] = {"name": name}
                if "ratio" in info:
                    entry["value"] = info["ratio"]
                elif "std" in info:
                    entry["value"] = info["std"]
                elif "laplacian_var" in info:
                    entry["value"] = info["laplacian_var"]
                elif "max_step" in info:
                    entry["value"] = info["max_step"]
                elif "range" in info:
                    entry["value"] = info["range"]
                if "expected" in info:
                    entry["threshold"] = str(info["expected"])
                failed.append(entry)

        if learned_conf is not None and learned_conf < 0.5:
            failed.append({
                "name": "learned_fundus_probability",
                "value": round(learned_conf, 4),
                "threshold": ">= 0.50",
            })

        return failed

    # ------------------------------------------------------------------
    # Visual evidence generation
    # ------------------------------------------------------------------

    def _generate_evidence(
        self,
        image: Image.Image,
        stat_checks: dict,
        learned_conf: float | None,
    ) -> dict[str, str]:
        """Generate base64-encoded diagnostic visualizations.

        Each visualization is independently try/excepted so one failure
        does not prevent others from being returned.
        """
        evidence: dict[str, str] = {}

        # 1. Radial gradient map
        try:
            evidence["radial_gradient_map"] = self._render_radial_gradient(image)
        except Exception as e:
            logger.debug("Failed to generate radial gradient map: %s", e)

        # 2. Green Laplacian heatmap
        try:
            evidence["green_laplacian_map"] = self._render_green_laplacian(image)
        except Exception as e:
            logger.debug("Failed to generate green Laplacian map: %s", e)

        # 3. Learned activation map (GradCAM)
        if self._learned_available and self._learned_gate is not None:
            try:
                evidence["learned_activation_map"] = self._render_gradcam(image)
            except Exception as e:
                logger.debug("Failed to generate GradCAM activation map: %s", e)

        return evidence

    @staticmethod
    def _render_radial_gradient(image: Image.Image) -> str:
        """Render radial luminance profile as a heatmap overlay."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        rgb = image.convert("RGB").resize((224, 224))
        pixels = np.array(rgb, dtype=np.float32) / 255.0
        luminance = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]

        h, w = luminance.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = min(h, w) / 2.0
        # Build radial distance map
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2) / max_r
        dist = np.clip(dist, 0, 1)

        fig = Figure(figsize=(3, 3), dpi=75)
        ax = fig.add_subplot(111)
        ax.imshow(pixels)
        ax.imshow(dist, alpha=0.5, cmap="hot")
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _render_green_laplacian(image: Image.Image) -> str:
        """Render green-channel Laplacian as a heatmap overlay."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        rgb = image.convert("RGB").resize((224, 224))
        pixels = np.array(rgb, dtype=np.float32) / 255.0
        green = pixels[:, :, 1]

        # Compute Laplacian (same kernel as fundus_gate.py)
        lap = (
            -4 * green[1:-1, 1:-1]
            + green[:-2, 1:-1] + green[2:, 1:-1]
            + green[1:-1, :-2] + green[1:-1, 2:]
        )
        lap_abs = np.abs(lap)
        # Normalize for visualization
        lap_norm = lap_abs / (lap_abs.max() + 1e-8)

        fig = Figure(figsize=(3, 3), dpi=75)
        ax = fig.add_subplot(111)
        ax.imshow(pixels)
        # Pad laplacian to match image size
        lap_padded = np.zeros_like(green)
        lap_padded[1:-1, 1:-1] = lap_norm
        ax.imshow(lap_padded, alpha=0.6, cmap="hot")
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _render_gradcam(self, image: Image.Image) -> str:
        """Generate GradCAM activation map from the learned gate."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        gate = self._learned_gate
        tensor = gate.transform(image.convert("RGB")).unsqueeze(0)

        # Try pytorch-grad-cam library (already a project dependency)
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            target_layer = self._gradcam_target_layer
            if target_layer is None:
                raise RuntimeError("No target layer identified for GradCAM")

            cam = GradCAM(model=gate.backbone, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

            rgb_np = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
            overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

            fig = Figure(figsize=(3, 3), dpi=75)
            ax = fig.add_subplot(111)
            ax.imshow(overlay)
            ax.set_axis_off()
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except ImportError:
            # Fallback: manual gradient-based saliency
            tensor.requires_grad_(True)
            gate.backbone.zero_grad()
            output = gate.backbone(tensor)
            output.backward()
            saliency = tensor.grad.data.abs().squeeze().mean(dim=0).numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            rgb_np = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0

            fig = Figure(figsize=(3, 3), dpi=75)
            ax = fig.add_subplot(111)
            ax.imshow(rgb_np)
            ax.imshow(saliency, alpha=0.5, cmap="jet")
            ax.set_axis_off()
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Module-level singleton (thread-safe lazy initialization)
# ---------------------------------------------------------------------------

_gate_v2: FundusGateV2 | None = None
_gate_v2_lock = threading.Lock()


def _get_gate() -> FundusGateV2:
    """Get or create the module-level FundusGateV2 singleton."""
    global _gate_v2
    if _gate_v2 is None:
        with _gate_v2_lock:
            if _gate_v2 is None:
                _gate_v2 = FundusGateV2()
    return _gate_v2


def gate_image(image: Image.Image) -> GateResultV2:
    """Run the full v2 fusion gate pipeline on an image.

    Drop-in replacement for src.data.fundus_gate.gate_image().
    Returns GateResultV2 (subclass of GateResult) — all existing
    attribute access (.passed, .confidence, .reason, .layer, .checks)
    works unchanged.
    """
    return _get_gate().gate_image(image)


# gate_predictions is re-exported from fundus_gate.py (imported at top of file)
__all__ = ["GateResultV2", "FundusGateV2", "gate_image", "gate_predictions"]

"""Gate router — status and debug endpoints for fundus gate v2."""
import io
import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gate", tags=["gate"])

# Global state — set during app startup (same pattern as agents.py)
_learned_gate = None
_gate_start_time: float = 0.0


def set_learned_gate(gate) -> None:
    """Called during app lifespan startup to inject the learned gate instance."""
    global _learned_gate, _gate_start_time
    _learned_gate = gate
    _gate_start_time = time.time()


def get_learned_gate():
    return _learned_gate


@router.get("/status")
async def gate_status():
    """Return gate version, thresholds, learned model info, and operational stats."""
    from src.monitoring.gate_monitor import gate_monitor

    metrics = gate_monitor.metrics()

    return {
        "gate_version": "v2",
        "enabled": settings.fundus_gate.enabled,
        "learned_model_loaded": _learned_gate is not None,
        "config": {
            "version": settings.fundus_gate.version,
            "learned_weight": settings.fundus_gate.learned_weight,
            "min_confidence": settings.fundus_gate.min_confidence,
            "model_path": settings.fundus_gate.model_path,
            "visual_evidence": settings.fundus_gate.visual_evidence,
        },
        "stats": {
            "total_checked": metrics.total_checked,
            "passed": metrics.passed,
            "rejected": metrics.rejected,
            "pass_rate": metrics.pass_rate,
            "rejection_by_layer": metrics.rejection_by_layer,
            "disagreements": metrics.learned_statistical_disagreements,
            "disagreement_rate": metrics.disagreement_rate,
        },
        "latency": {
            "p50_ms": metrics.latency_p50_ms,
            "p95_ms": metrics.latency_p95_ms,
            "p99_ms": metrics.latency_p99_ms,
        },
        "alert": {
            "active": metrics.alert_active,
            "message": metrics.alert_message,
        },
        "uptime_seconds": round(gate_monitor.uptime_seconds, 1),
    }


@router.post("/validate")
async def gate_validate(file: UploadFile = File(...)):
    """Debug endpoint — run full gate pipeline and return detailed breakdown.

    Always returns 200 (never 422). Designed for clinicians and ops to
    test gate behavior on specific images without triggering model inference.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG)")

    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size // 1024 // 1024}MB)")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
    except (OSError, SyntaxError) as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    # Always run with visual evidence enabled for the debug endpoint
    from src.data.fundus_gate_v2 import FundusGateV2

    debug_gate = FundusGateV2(
        enabled=True,
        learned_weight=settings.fundus_gate.learned_weight,
        min_confidence=settings.fundus_gate.min_confidence,
        model_path=settings.fundus_gate.model_path,
        visual_evidence=True,
    )

    result = debug_gate.gate_image(image)

    return {
        "passed": result.passed,
        "confidence": round(result.confidence, 4),
        "layer": result.layer,
        "reason": result.reason,
        "gate_version": result.gate_version,
        "statistical_confidence": round(result.statistical_confidence, 4),
        "learned_confidence": (
            round(result.learned_confidence, 4) if result.learned_confidence is not None else None
        ),
        "fused_confidence": round(result.fused_confidence, 4),
        "fusion_weights": result.fusion_weights,
        "suggested_action": result.suggested_action,
        "failed_checks": result.failed_checks,
        "checks": result.checks,
        "visual_evidence": result.visual_evidence,
        "latency_ms": round(result.latency_ms, 2),
        "image_size": {"width": image.width, "height": image.height},
    }

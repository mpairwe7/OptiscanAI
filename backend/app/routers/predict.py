"""Prediction router - image upload, fundus gating, and inference."""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile, HTTPException, Query
from PIL import Image
import io

from backend.app.core.model_service import model_service
from backend.app.core.config import settings
from backend.app.core.prediction_logger import prediction_logger
from backend.app.core.auth import get_current_user, TokenPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    request: Request = None,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict retinal diseases from uploaded fundus image.

    Images pass through a 3-layer fundus validation gate:
    1. Structural: format, resolution, aspect ratio, color mode
    2. Statistical: channel histograms, red dominance, dark border, center brightness
    3. Post-inference OOD: flags results if model confidence is near-zero across all diseases
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG)")

    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size // 1024 // 1024}MB)")

    # Validate image magic bytes (JPEG: FFD8FF, PNG: 89504E47)
    if len(contents) < 4:
        raise HTTPException(400, "File too small to be a valid image")
    header = contents[:4]
    if not (header[:2] == b'\xff\xd8' or header[:4] == b'\x89PNG'):
        raise HTTPException(400, "Unsupported image format. Only JPEG and PNG are accepted.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Check structural integrity
        image = Image.open(io.BytesIO(contents))  # Re-open after verify
    except (OSError, SyntaxError) as e:
        logger.warning("Invalid image upload: %s", str(e))
        raise HTTPException(400, f"Invalid image file: {e}")

    if image.width < 32 or image.height < 32:
        raise HTTPException(400, "Image too small (min 32x32)")

    # ── Fundus image gating (pre-inference) ──
    from src.data.fundus_gate_v2 import gate_image, gate_predictions

    gate_result = gate_image(image)
    if not gate_result.passed:
        logger.info(
            "Image rejected by fundus gate v2 (%s): %s",
            gate_result.layer,
            gate_result.reason,
        )
        # Log rejection for audit trail
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())) if request else str(uuid.uuid4())
        prediction_logger.log_gate_rejection(
            request_id=request_id,
            user=user.sub if user else "anonymous",
            gate_result=gate_result,
            image_size=(image.width, image.height),
        )
        # Record in gate monitor
        from src.monitoring.gate_monitor import gate_monitor
        gate_monitor.record(
            passed=False,
            layer=gate_result.layer,
            latency_ms=getattr(gate_result, "latency_ms", 0),
            statistical_passed=False,
            learned_passed=None,
        )
        detail = {
            "error": "non_fundus_image",
            "message": gate_result.reason,
            "confidence": gate_result.confidence,
            "layer": gate_result.layer,
            "checks": gate_result.checks,
        }
        # V2 fields
        if hasattr(gate_result, "fundus_confidence"):
            detail["fundus_confidence"] = gate_result.fused_confidence
        if hasattr(gate_result, "failed_checks") and gate_result.failed_checks:
            detail["failed_checks"] = gate_result.failed_checks
        if hasattr(gate_result, "visual_evidence") and gate_result.visual_evidence:
            detail["visual_evidence"] = gate_result.visual_evidence
        if hasattr(gate_result, "suggested_action") and gate_result.suggested_action:
            detail["suggestion"] = gate_result.suggested_action
        if hasattr(gate_result, "fusion_weights"):
            detail["fusion_weights"] = gate_result.fusion_weights
        raise HTTPException(422, detail=detail)

    # ── Model inference ──
    result = model_service.predict(image, threshold=threshold)

    # ── Post-inference OOD check ──
    ood_result = gate_predictions(
        result.get("predictions", []),
        result.get("all_probabilities", {}),
    )
    if ood_result is not None:
        result["ood_warning"] = {
            "flagged": True,
            "message": ood_result.reason,
            "checks": ood_result.checks,
        }
        logger.warning("OOD warning for request: %s", ood_result.reason)

    # Add gate metadata to response
    result["fundus_gate"] = {
        "passed": gate_result.passed,
        "confidence": round(gate_result.confidence, 3),
    }
    if hasattr(gate_result, "gate_version"):
        result["fundus_gate"]["version"] = gate_result.gate_version
        result["fundus_gate"]["latency_ms"] = round(gate_result.latency_ms, 2)
        result["fundus_gate"]["statistical_confidence"] = round(gate_result.statistical_confidence, 4)
        result["fundus_gate"]["learned_confidence"] = (
            round(gate_result.learned_confidence, 4)
            if gate_result.learned_confidence is not None
            else None
        )
        result["fundus_gate"]["fusion_confidence"] = round(gate_result.fused_confidence, 4)

    # Record gate pass in monitor
    from src.monitoring.gate_monitor import gate_monitor
    gate_monitor.record(
        passed=True,
        layer=gate_result.layer,
        latency_ms=getattr(gate_result, "latency_ms", 0),
        statistical_passed=True,
        learned_passed=(
            gate_result.learned_confidence > 0.5
            if hasattr(gate_result, "learned_confidence") and gate_result.learned_confidence is not None
            else None
        ),
    )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())) if request else str(uuid.uuid4())

    prediction_logger.log(
        request_id=request_id,
        user=user.sub if user else "anonymous",
        predictions=result.get("predictions", []),
        threshold=result.get("threshold"),
        threshold_source=result.get("threshold_source", "scalar"),
        inference_ms=result.get("inference_ms", 0),
        model_loaded=result.get("model_loaded", False),
        image_size=(image.width, image.height),
        num_detected=result.get("total_detected", 0),
        referral_priority=result.get("clinical", {}).get("referral_priority", ""),
        fundus_gate_version=getattr(gate_result, "gate_version", "v1"),
        learned_score=getattr(gate_result, "learned_confidence", -1.0) or -1.0,
        statistical_score=getattr(gate_result, "statistical_confidence", -1.0),
        fusion_confidence=getattr(gate_result, "fused_confidence", -1.0),
    )

    # Notify agents of the new scan
    try:
        from src.agents.event_bus import event_bus, Event, EventType
        await event_bus.emit(Event(
            type=EventType.SCAN_ANALYZED,
            source="predict_endpoint",
            data={
                "scan_id": request_id,
                "diseases_detected": result.get("total_detected", 0),
                "referral_priority": result.get("clinical", {}).get("referral_priority", ""),
                "inference_ms": result.get("inference_ms", 0),
                "needs_review": result.get("total_detected", 0) > 5,
            },
        ))
    except Exception:
        pass  # agents are optional

    return {"success": True, "request_id": request_id, **result}


@router.get("/diseases")
async def list_diseases():
    """List all detectable diseases."""
    from backend.app.core.model_service import DISEASE_NAMES
    return {
        "total": len(model_service.disease_codes),
        "diseases": [
            {"code": code, "name": DISEASE_NAMES.get(code, code)}
            for code in model_service.disease_codes
        ],
    }

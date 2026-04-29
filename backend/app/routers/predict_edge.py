"""Edge prediction API endpoints for ONNX, CoreML, and quantized inference.

Provides the same prediction interface as the main ``/api/v1/predict``
endpoint but routes inference through the ``EdgeRuntime`` instead of
the standard ``ModelService``.  Each format returns 501 if the
corresponding model has not been loaded.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["edge-inference"])

# ---------------------------------------------------------------------------
# Lazy singleton -- initialised once on first request
# ---------------------------------------------------------------------------

_edge_runtime = None


def _get_runtime():
    """Lazy-initialise the EdgeRuntime singleton.

    Models are loaded based on ``settings.edge.*_enabled`` flags.  If no
    edge format is enabled the runtime is still created (but all predict
    calls will return 501).
    """
    global _edge_runtime
    if _edge_runtime is not None:
        return _edge_runtime

    from src.serving.edge_runtime import EdgeRuntime

    _edge_runtime = EdgeRuntime()

    cfg = settings.edge

    if cfg.onnx_enabled:
        try:
            _edge_runtime.load_onnx(cfg.onnx_model_path)
        except Exception:
            logger.warning(
                "ONNX model failed to load from %s",
                cfg.onnx_model_path,
                exc_info=True,
            )

    if cfg.coreml_enabled:
        try:
            _edge_runtime.load_coreml(cfg.coreml_model_path)
        except Exception:
            logger.warning(
                "CoreML model failed to load from %s",
                cfg.coreml_model_path,
                exc_info=True,
            )

    if cfg.quantized_enabled:
        try:
            _edge_runtime.load_quantized(cfg.quantized_model_path)
        except Exception:
            logger.warning(
                "Quantized model failed to load from %s",
                cfg.quantized_model_path,
                exc_info=True,
            )

    logger.info(
        "EdgeRuntime initialised -- loaded formats: %s",
        _edge_runtime.get_loaded_formats(),
    )
    return _edge_runtime


# ---------------------------------------------------------------------------
# Shared image validation
# ---------------------------------------------------------------------------

async def _validate_and_open(file: UploadFile) -> Image.Image:
    """Read, validate, and return a PIL Image from an upload.

    Raises ``HTTPException`` on invalid input.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG/PNG).",
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_size // 1024 // 1024}MB).",
        )

    # Magic-byte validation
    if len(contents) < 4:
        raise HTTPException(status_code=400, detail="File too small to be a valid image.")
    header = contents[:4]
    if not (header[:2] == b"\xff\xd8" or header[:4] == b"\x89PNG"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Only JPEG and PNG are accepted.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  # re-open after verify
    except (OSError, SyntaxError) as exc:
        logger.warning("Invalid image upload: %s", exc)
        raise HTTPException(
            status_code=400, detail=f"Invalid image file: {exc}"
        )

    if image.width < 32 or image.height < 32:
        raise HTTPException(
            status_code=400, detail="Image too small (minimum 32x32)."
        )

    return image


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/onnx")
async def predict_onnx(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Run inference using the ONNX Runtime backend.

    Returns the same prediction schema as ``/api/v1/predict`` with an
    additional ``runtime_format`` field set to ``"onnx"``.
    """
    runtime = _get_runtime()

    if "onnx" not in runtime.get_loaded_formats():
        raise HTTPException(
            status_code=501,
            detail="ONNX model is not loaded. Enable via EDGE__ONNX_ENABLED=true.",
        )

    image = await _validate_and_open(file)

    try:
        result = runtime.predict_onnx(image, threshold=threshold)
    except Exception as exc:
        logger.exception("ONNX prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"ONNX inference error: {exc}",
        )

    return {"success": True, **result}


@router.post("/coreml")
async def predict_coreml(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Run inference using the CoreML backend (Apple Silicon).

    Returns the same prediction schema as ``/api/v1/predict`` with an
    additional ``runtime_format`` field set to ``"coreml"``.
    """
    runtime = _get_runtime()

    loaded = runtime.get_loaded_formats()
    if not any(f.startswith("coreml") for f in loaded):
        raise HTTPException(
            status_code=501,
            detail="CoreML model is not loaded. Enable via EDGE__COREML_ENABLED=true.",
        )

    image = await _validate_and_open(file)

    try:
        result = runtime.predict_coreml(image, threshold=threshold)
    except Exception as exc:
        logger.exception("CoreML prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"CoreML inference error: {exc}",
        )

    return {"success": True, **result}


@router.post("/quantized")
async def predict_quantized(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    precision: str = Query("int8", description="Quantization precision (int8, fp16)"),
):
    """Run inference using a quantized PyTorch model.

    The ``precision`` query parameter selects the quantization flavour.
    Returns the same prediction schema as ``/api/v1/predict`` with an
    additional ``runtime_format`` field.
    """
    runtime = _get_runtime()

    loaded = runtime.get_loaded_formats()
    if not any(f.startswith("quantized") for f in loaded):
        raise HTTPException(
            status_code=501,
            detail=(
                "Quantized model is not loaded. "
                "Enable via EDGE__QUANTIZED_ENABLED=true."
            ),
        )

    image = await _validate_and_open(file)

    try:
        result = runtime.predict_quantized(image, threshold=threshold)
    except Exception as exc:
        logger.exception("Quantized prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Quantized inference error: {exc}",
        )

    result["precision"] = precision
    return {"success": True, **result}


@router.get("/edge/status")
async def edge_status():
    """Return which edge inference formats are currently loaded."""
    runtime = _get_runtime()
    loaded = runtime.get_loaded_formats()

    return {
        "loaded_formats": loaded,
        "onnx_enabled": settings.edge.onnx_enabled,
        "coreml_enabled": settings.edge.coreml_enabled,
        "quantized_enabled": settings.edge.quantized_enabled,
        "parity_tolerance": settings.edge.parity_tolerance,
    }

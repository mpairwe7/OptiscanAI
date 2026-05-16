"""Explainability endpoints — GradCAM, SHAP, LIME, Integrated Gradients."""

import base64
import io
import logging
import time
from typing import Optional

import torch
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.feature_gate import require_tier
from backend.app.core.model_service import DISEASE_NAMES, model_service

# Clinician-or-above gate for advanced XAI methods.
_clinician = require_tier("clinician", feature="advanced_explainability")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])

# Lazy-initialized explainer singleton
_explainer = None


def _get_explainer():
    """Get or create ModelExplainer singleton."""
    global _explainer
    if _explainer is not None:
        return _explainer
    if not model_service.is_loaded:
        raise HTTPException(503, "Model not loaded")

    from src.models.model_explainer import ModelExplainer

    _explainer = ModelExplainer(
        model=model_service.model,
        device=str(model_service.device),
        disease_names=model_service.disease_codes,
        mobile_mode=False,
    )
    return _explainer


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to preprocessed model input tensor."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0).to(model_service.device)


def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image as base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


async def _read_upload(file: UploadFile) -> Image.Image:
    """Read and validate uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG)")
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            413, f"File too large (max {settings.max_upload_size // 1024 // 1024}MB)"
        )
    try:
        return Image.open(io.BytesIO(contents))
    except (OSError, SyntaxError) as e:
        logger.warning("Invalid image in explainability upload: %s", e)
        raise HTTPException(400, "Invalid image file")


@router.post("/gradcam")
async def explain_gradcam(
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(
        None, description="Target class index (auto-selects top prediction if omitted)"
    ),
    method: str = Query("GradCAM", description="CAM variant: GradCAM, GradCAMPlusPlus, ScoreCAM"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top classes to explain"),
):
    """Generate GradCAM heatmap overlay for uploaded image.

    Returns base64-encoded heatmap images for each target class."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()

    # If specific class requested, generate single heatmap image
    if target_class is not None:
        try:
            heatmap_pil = explainer.generate_gradcam(tensor, target_class=target_class)
            elapsed = (time.perf_counter() - t0) * 1000
            disease_code = (
                model_service.disease_codes[target_class]
                if target_class < len(model_service.disease_codes)
                else f"class_{target_class}"
            )
            return {
                "method": method,
                "target_class": target_class,
                "disease": disease_code,
                "disease_name": DISEASE_NAMES.get(disease_code, disease_code),
                "heatmap": _pil_to_base64(heatmap_pil),
                "original": _pil_to_base64(image.resize((224, 224))),
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as e:
            raise HTTPException(500, f"GradCAM failed: {str(e)}")

    # Multi-class: get CAM data for top-K predictions
    with torch.no_grad():
        output = model_service.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.sigmoid(output).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1].tolist()

    heatmaps = []
    for idx in top_indices:
        try:
            heatmap_pil = explainer.generate_gradcam(tensor, target_class=idx)
            code = model_service.disease_codes[idx]
            heatmaps.append(
                {
                    "class_index": idx,
                    "disease": code,
                    "disease_name": DISEASE_NAMES.get(code, code),
                    "probability": round(float(probs[idx]), 4),
                    "heatmap": _pil_to_base64(heatmap_pil),
                }
            )
        except Exception as e:
            logger.warning(f"GradCAM failed for class {idx}: {e}")
            code = model_service.disease_codes[idx]
            heatmaps.append(
                {
                    "class_index": idx,
                    "disease": code,
                    "disease_name": DISEASE_NAMES.get(code, code),
                    "probability": round(float(probs[idx]), 4),
                    "heatmap": None,
                    "error": str(e),
                }
            )

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "method": method,
        "original": _pil_to_base64(image.resize((224, 224))),
        "heatmaps": heatmaps,
        "elapsed_ms": round(elapsed, 2),
    }


@router.post("/lime")
async def explain_lime(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=5, description="Number of classes to explain"),
    num_samples: int = Query(
        300, ge=50, le=2000, description="Perturbation samples (higher=slower but more accurate)"
    ),
    num_features: int = Query(10, ge=3, le=30, description="Number of superpixels"),
    _gate=Depends(_clinician),
):
    """Generate LIME superpixel explanations for uploaded image."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()

    # Get top predictions
    with torch.no_grad():
        output = model_service.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.sigmoid(output).cpu().numpy()[0]

    top_indices = [int(i) for i in probs.argsort()[-top_k:][::-1]]

    result = explainer.explain_lime(
        tensor, target_classes=top_indices, num_samples=num_samples, num_features=num_features
    )
    elapsed = (time.perf_counter() - t0) * 1000

    # Simplify the response — convert numpy types to native Python for JSON serialization
    simplified = {}
    for disease_name, data in result.items():
        if "error" in data:
            simplified[disease_name] = {"error": str(data["error"])}
        else:
            # Convert numpy keys/values to native Python types
            raw_weights = data.get("feature_weights", {})
            clean_weights = {str(k): float(v) for k, v in raw_weights.items()}
            simplified[disease_name] = {
                "prediction": float(data.get("prediction", 0)),
                "segments": int(data.get("lime_segments", num_features)),
                "samples_used": int(data.get("samples_used", num_samples)),
                "summary": {
                    str(k): float(v) if isinstance(v, (int, float)) else v
                    for k, v in data.get("explanation_summary", {}).items()
                },
                "feature_weights": clean_weights,
            }

    return {
        "method": "LIME",
        "explanations": simplified,
        "elapsed_ms": round(elapsed, 2),
    }


@router.post("/shap")
async def explain_shap(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=5, description="Number of classes to explain"),
    _gate=Depends(_clinician),
):
    """Generate SHAP feature importance explanations."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()

    with torch.no_grad():
        output = model_service.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.sigmoid(output).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1].tolist()

    result = explainer.explain_shap(tensor, target_classes=top_indices)
    elapsed = (time.perf_counter() - t0) * 1000

    # Simplify — don't send raw shap arrays
    simplified = {}
    for disease_name, data in result.items():
        if "error" in data:
            simplified[disease_name] = {"error": data["error"]}
        else:
            simplified[disease_name] = {
                "prediction": data.get("prediction", 0),
                "feature_importance": data.get("feature_importance", {}),
            }

    return {
        "method": "SHAP",
        "explanations": simplified,
        "elapsed_ms": round(elapsed, 2),
    }


@router.post("/integrated-gradients")
async def explain_integrated_gradients(
    file: UploadFile = File(...),
    top_k: int = Query(2, ge=1, le=5),
    n_steps: int = Query(25, ge=5, le=100),
    _gate=Depends(_clinician),
):
    """Generate Integrated Gradients attribution maps."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()

    with torch.no_grad():
        output = model_service.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.sigmoid(output).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1].tolist()

    result = explainer.explain_integrated_gradients(
        tensor, target_classes=top_indices, n_steps=n_steps
    )
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "method": "IntegratedGradients",
        "explanations": result,
        "elapsed_ms": round(elapsed, 2),
    }


@router.post("/eli5")
async def explain_eli5(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=5),
    top_features: int = Query(10, ge=3, le=20),
    _gate=Depends(_clinician),
):
    """Generate ELI5 human-readable explanations with feature importance."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()

    with torch.no_grad():
        output = model_service.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.sigmoid(output).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1].tolist()

    result = explainer.explain_eli5(tensor, target_classes=top_indices, top_features=top_features)
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "method": "ELI5",
        "explanations": result,
        "elapsed_ms": round(elapsed, 2),
    }


@router.post("/comprehensive")
async def explain_comprehensive(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
    _gate=Depends(_clinician),
):
    """Run all available explainability methods on uploaded image.

    Returns GradCAM heatmaps + clinical insights + uncertainty metrics +
    any available LIME/SHAP/IG results."""
    explainer = _get_explainer()
    image = await _read_upload(file)
    tensor = _image_to_tensor(image)

    t0 = time.perf_counter()
    result = explainer.get_lightweight_explanation(tensor, top_k=top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    # Generate base64 heatmaps for the top predictions
    gradcam_images = []
    if result.get("predictions"):
        for pred in result["predictions"][:3]:
            disease_short = pred.get("disease", "")
            # Find class index
            try:
                idx = (
                    model_service.disease_codes.index(disease_short)
                    if disease_short in model_service.disease_codes
                    else None
                )
            except ValueError:
                idx = None

            if idx is not None:
                try:
                    heatmap_pil = explainer.generate_gradcam(tensor, target_class=idx)
                    gradcam_images.append(
                        {
                            "disease": disease_short,
                            "disease_name": DISEASE_NAMES.get(disease_short, disease_short),
                            "probability": pred.get("confidence_score", 0),
                            "heatmap": _pil_to_base64(heatmap_pil),
                        }
                    )
                except Exception as e:
                    logger.warning("Heatmap generation failed for class %s: %s", idx, e)

    result["gradcam_heatmaps"] = gradcam_images
    result["original_image"] = _pil_to_base64(image.resize((224, 224)))
    result["elapsed_ms"] = round(elapsed, 2)

    # Strip large array fields from explainability to keep response reasonable
    if "explainability" in result:
        expl = result["explainability"]
        # Strip raw CAM arrays from GradCAM (keep prediction scores only)
        if "gradcam" in expl and isinstance(expl["gradcam"], dict):
            for disease, data in expl["gradcam"].items():
                if isinstance(data, dict) and "cam" in data:
                    del data["cam"]
        # Strip raw arrays from LIME
        if "lime" in expl and isinstance(expl["lime"], dict):
            for disease, data in expl["lime"].items():
                if isinstance(data, dict):
                    data.pop("explained_image", None)
                    data.pop("mask", None)
        # Strip raw arrays from SHAP
        if "shap" in expl and isinstance(expl["shap"], dict):
            for disease, data in expl["shap"].items():
                if isinstance(data, dict):
                    data.pop("shap_values", None)
                    data.pop("shap_magnitude", None)
                    data.pop("shap_normalized", None)

    return result


@router.get("/available")
async def get_available_methods():
    """List which explainability methods are available in the current environment."""
    from src.models.model_explainer import (
        CAPTUM_AVAILABLE,
        ELI5_AVAILABLE,
        GRADCAM_AVAILABLE,
        LIME_AVAILABLE,
        SHAP_AVAILABLE,
    )

    return {
        "model_loaded": model_service.is_loaded,
        "methods": {
            "gradcam": {
                "available": GRADCAM_AVAILABLE,
                "description": "Gradient-weighted Class Activation Mapping — highlights image regions most relevant to each prediction",
            },
            "integrated_gradients": {
                "available": CAPTUM_AVAILABLE,
                "description": "Attribution method from Captum — pixel-level importance scores",
            },
            "shap": {
                "available": SHAP_AVAILABLE,
                "description": "SHapley Additive exPlanations — game-theoretic feature importance",
            },
            "lime": {
                "available": LIME_AVAILABLE,
                "description": "Local Interpretable Model-agnostic Explanations — superpixel perturbation analysis",
            },
            "eli5": {
                "available": ELI5_AVAILABLE,
                "description": "Explain Like I'm 5 — human-readable feature importance",
            },
        },
    }

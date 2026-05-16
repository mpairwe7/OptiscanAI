"""Analytics, system info, and reporting endpoints."""

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.core.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/system/info")
async def system_info():
    """Deployment and system information for the platform dashboard."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_memory = None
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        gpu_memory = f"{mem / (1024**3):.1f} GB"

    return {
        "platform": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "region": settings.deployment_region,
            "regulatory_mode": settings.regulatory_mode,
        },
        "model": {
            "name": settings.model_name,
            "loaded": model_service.is_loaded,
            "num_classes": settings.num_classes,
            "diseases_covered": len(model_service.disease_codes),
            "knowledge_graph_edges": model_service.kg.get_edge_count() if model_service.kg else 0,
            "threshold_source": (
                "per_class" if model_service.default_thresholds is not None else "scalar"
            ),
        },
        "infrastructure": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu": gpu_name,
            "gpu_memory": gpu_memory,
            "device": str(model_service.device) if model_service.device else "not initialized",
        },
        "capabilities": {
            "explainability_methods": ["GradCAM", "LIME", "SHAP", "Integrated Gradients", "ELI5"],
            "clinical_reasoning": True,
            "knowledge_graph": model_service.kg is not None,
            "human_review": True,
            "audit_trail": True,
            "drift_detection": True,
        },
        "compliance": {
            "eu_ai_act": "conformity_ready",
            "fda_samd": "pre_submission",
            "data_governance": True,
            "model_cards": True,
            "fairness_evaluation": True,
            "prediction_logging": True,
        },
    }


@router.get("/analytics/summary")
async def analytics_summary():
    """Get prediction analytics summary from logged predictions."""
    log_dir = Path(settings.prediction_log_dir)
    if not log_dir.exists():
        return {
            "total_scans": 0,
            "today_scans": 0,
            "avg_inference_ms": 0,
            "referral_distribution": {},
            "top_detected_diseases": [],
            "daily_volumes": [],
        }

    total_scans = 0
    today_scans = 0
    inference_times = []
    referral_counts: dict[str, int] = {}
    disease_counts: dict[str, int] = {}
    daily_volumes: dict[str, int] = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for log_file in sorted(log_dir.glob("predictions_*.jsonl"))[-30:]:
        date_str = log_file.stem.replace("predictions_", "")
        day_count = 0
        for line in log_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                total_scans += 1
                day_count += 1
                if date_str == today:
                    today_scans += 1
                inference_times.append(entry.get("inference_ms", 0))
                priority = entry.get("referral_priority", "UNKNOWN")
                referral_counts[priority] = referral_counts.get(priority, 0) + 1
                for pred in entry.get("top_predictions", []):
                    code = pred.get("code", "")
                    if code:
                        disease_counts[code] = disease_counts.get(code, 0) + 1
            except json.JSONDecodeError:
                continue
        daily_volumes[date_str] = day_count

    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0

    return {
        "total_scans": total_scans,
        "today_scans": today_scans,
        "avg_inference_ms": round(avg_inference, 2),
        "referral_distribution": referral_counts,
        "top_detected_diseases": [{"code": code, "count": count} for code, count in top_diseases],
        "daily_volumes": [{"date": d, "scans": c} for d, c in list(daily_volumes.items())[-14:]],
    }

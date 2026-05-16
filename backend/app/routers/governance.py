"""
Governance API endpoints for drift monitoring, active learning, model registry,
fairness evaluation, and model card access.

Phase 1: /drift, /active-learning-stats, /model-registry
Phase 3: /fairness, /fairness/history, /model-card, /audit, /audit/integrity
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/governance",
    tags=["governance"],
    dependencies=[Depends(require_tier("practice", feature="governance"))],
)


# ── Phase 1 Endpoints ───────────────────────────────────────────────────────


@router.get("/drift")
async def get_drift_status(
    include_history: bool = Query(False, description="Include historical drift checks"),
    history_limit: int = Query(50, ge=1, le=500, description="Max history entries"),
) -> dict:
    """Current drift detection status with optional history.

    Returns results from all enabled detectors: PSI, KS-test, confidence drop,
    and optionally NannyML CBPE and Evidently DataDrift.
    """
    from backend.app.core.drift_detector import get_drift_detector

    detector = get_drift_detector()
    if detector is None:
        return {
            "enabled": False,
            "message": "Drift detection not initialized",
            "current": None,
            "history": None,
        }

    current = detector.get_current_status()
    history = detector.get_history(limit=history_limit) if include_history else None

    return {
        "enabled": True,
        "current": current,
        "history": history,
        "detectors_enabled": {
            "psi": True,
            "ks_test": True,
            "confidence_drop": True,
            "nannyml": settings.drift.nannyml_enabled,
            "evidently": settings.drift.evidently_enabled,
        },
    }


@router.get("/active-learning-stats")
async def get_active_learning_stats() -> dict:
    """Active learning queue statistics and fine-tuning history.

    Shows progress toward retrain threshold, corrected sample counts,
    and history of fine-tuning runs.
    """
    from backend.app.core.active_learning import get_active_learning_loop

    loop = get_active_learning_loop()
    if loop is None:
        return {
            "enabled": False,
            "message": "Active learning loop not initialized",
        }

    return loop.get_stats()


@router.get("/model-registry")
async def get_model_registry_status() -> dict:
    """MLflow model registry status.

    Shows current production model version, staging versions,
    active shadow deployments, and experiment info.
    """
    from backend.app.core.mlflow_registry import get_mlflow_registry

    registry = get_mlflow_registry()
    if registry is None:
        return {
            "enabled": False,
            "message": "MLflow registry not initialized",
        }

    return registry.get_registry_status()


# ── Phase 3 Endpoints ───────────────────────────────────────────────────────


@router.get("/fairness")
async def get_fairness_report(
    model_version: Optional[str] = Query(None, description="Model version to evaluate"),
) -> dict:
    """Fairness dashboard data with demographic breakdowns.

    Returns performance metrics across protected attributes:
    age group, sex, ethnicity, camera device, and geography.
    """
    if not settings.fairness.enabled:
        return {
            "enabled": False,
            "message": "Fairness dashboard not enabled (FAIRNESS__ENABLED=false)",
        }

    try:
        from src.governance.fairness import FairnessEvaluator

        evaluator = FairnessEvaluator()

        # Category-level fairness (always available without demographic data)
        category_report = evaluator.evaluate_category_fairness()

        return {
            "enabled": True,
            "model_version": model_version or "current",
            "category_fairness": category_report,
            "demographic_breakdown": {
                attr: {"status": "no_data", "message": "Demographic metadata required"}
                for attr in settings.fairness.protected_attributes
            },
            "recommendations": [
                "Collect demographic metadata to enable full fairness analysis",
                "Run bias audit with scripts/run_bias_audit.py for detailed report",
            ],
        }
    except Exception as e:
        logger.error(f"Fairness evaluation failed: {e}", exc_info=True)
        return {"enabled": True, "error": str(e)}


@router.get("/fairness/history")
async def get_fairness_history(
    limit: int = Query(30, ge=1, le=100),
) -> dict:
    """Historical fairness evaluation trends."""
    if not settings.fairness.enabled:
        return {"enabled": False, "history": []}

    return {
        "enabled": True,
        "history": [],
        "message": "Historical tracking requires fairness evaluations to be run periodically",
    }


@router.get("/model-card")
async def get_model_card(
    model_version: Optional[str] = Query(None),
    format: str = Query("json", pattern="^(json|markdown)$"),
) -> dict:
    """Current model card for the active or specified model version."""
    try:
        from src.governance.model_card import ModelCard

        card = ModelCard(
            model_name=settings.model_name,
            model_version=model_version or settings.app_version,
            task="Multi-label retinal disease classification",
            num_classes=settings.num_classes,
        )

        if format == "markdown":
            return {"format": "markdown", "content": card.to_markdown()}

        return {"format": "json", "card": card.to_dict()}
    except Exception as e:
        logger.error(f"Model card generation failed: {e}", exc_info=True)
        return {"error": str(e)}


@router.get("/audit")
async def query_audit_log(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """Query audit log entries.

    Returns recent audit events from the immutable audit trail.
    """
    try:
        from src.governance.audit import AuditTrail

        trail = AuditTrail()
        entries = trail.get_recent(limit=limit, event_type=event_type)

        return {
            "total": len(entries),
            "entries": entries,
        }
    except Exception as e:
        logger.error(f"Audit log query failed: {e}", exc_info=True)
        return {"error": str(e), "entries": []}


@router.get("/audit/integrity")
async def verify_audit_integrity() -> dict:
    """Verify audit log chain integrity via SHA-256 checksums."""
    try:
        from src.governance.audit import AuditTrail

        trail = AuditTrail()
        result = trail.verify_integrity()

        return {
            "integrity_verified": result.get("valid", False),
            "details": result,
        }
    except Exception as e:
        logger.error(f"Audit integrity check failed: {e}", exc_info=True)
        return {"integrity_verified": False, "error": str(e)}

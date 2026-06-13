"""Health, monitoring, and metadata router."""

from dataclasses import asdict

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.model_service import model_service
from src.monitoring.health import HealthMonitor

router = APIRouter(tags=["health"])

# Global health monitor
health_monitor = HealthMonitor(max_latency_p99_ms=100.0, max_error_rate=0.05)


@router.get("/health")
async def health():
    """Liveness probe — always 200 while the process is up."""
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
        "device": str(model_service.device) if model_service.device else "not initialized",
        "diseases_count": len(model_service.disease_codes),
    }


@router.get("/health/ready")
async def readiness():
    """Readiness probe — 200 only once the model is loaded, else 503.

    The container HEALTHCHECK and Crane Cloud readiness probe point here so
    traffic isn't routed until inference is actually possible.
    """
    if not model_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model_loaded": False},
        )
    return {"status": "ready", "model_loaded": True}


@router.get("/health/model")
async def model_health():
    """Detailed model health metrics: latency, throughput, SLA."""
    report = health_monitor.report()
    return asdict(report)


@router.get("/health/gate")
async def gate_health():
    """Fundus gate v2 operational metrics."""
    from dataclasses import asdict as _asdict

    from src.monitoring.gate_monitor import gate_monitor

    return _asdict(gate_monitor.metrics())


@router.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": model_service.is_loaded,
        "gpu_available": torch.cuda.is_available(),
        "diseases": len(model_service.disease_codes),
        "docs": "/docs",
    }

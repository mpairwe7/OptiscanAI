"""Health, monitoring, and metadata router."""
import torch
from dataclasses import asdict
from fastapi import APIRouter
from backend.app.core.model_service import model_service
from backend.app.core.config import settings
from src.monitoring.health import HealthMonitor

router = APIRouter(tags=["health"])

# Global health monitor
health_monitor = HealthMonitor(max_latency_p99_ms=100.0, max_error_rate=0.05)


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
        "device": str(model_service.device) if model_service.device else "not initialized",
        "diseases_count": len(model_service.disease_codes),
    }


@router.get("/health/model")
async def model_health():
    """Detailed model health metrics: latency, throughput, SLA."""
    report = health_monitor.report()
    return asdict(report)


@router.get("/health/gate")
async def gate_health():
    """Fundus gate v2 operational metrics."""
    from src.monitoring.gate_monitor import gate_monitor
    from dataclasses import asdict as _asdict
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

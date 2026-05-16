"""
FastAPI application entry point with modern lifespan management.
UV-managed, production-ready retinal disease classification API.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response

from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.core.model_service import model_service
from backend.app.routers import (
    agents,
    analytics,
    auth,
    billing,
    clinical,
    explain,
    gate,
    governance,
    health,
    monitoring,
    offline,
    orgs,
    predict,
    predict_edge,
    quantized,
    review,
)

setup_logging(settings.log_level, settings.log_format)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + Phase 1-4 services + agents. Shutdown in reverse."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # ── Stage -1: Database (must be up before auth/billing/quota) ──
    if settings.database.enabled:
        try:
            from backend.app.core.db import init_engine
            init_engine()
            logger.info("Database engine initialized")
        except Exception as e:
            logger.error(f"Database init failed (non-fatal): {e}", exc_info=True)

    # ── Stage 0: OpenTelemetry (first, so everything else is traced) ──
    if settings.telemetry.enabled:
        try:
            from backend.app.core.telemetry import init_telemetry
            init_telemetry(app)
            logger.info("OpenTelemetry initialized")
        except Exception as e:
            logger.error(f"OpenTelemetry init failed (non-fatal): {e}", exc_info=True)

    # ── Stage 1: Model loading ──
    model_service.load()

    # ── Stage 1.5: MLflow Model Registry (Phase 1) ──
    if settings.mlflow.enabled:
        try:
            from backend.app.core.mlflow_registry import init_mlflow_registry
            init_mlflow_registry()
            logger.info("MLflow registry initialized")
        except Exception as e:
            logger.error(f"MLflow init failed (non-fatal): {e}", exc_info=True)

    # ── Stage 2: Active Learning Loop (Phase 1) ──
    if settings.active_learning_loop.enabled:
        try:
            from backend.app.core.active_learning import init_active_learning
            init_active_learning()
            logger.info("Active learning loop initialized")
        except Exception as e:
            logger.error(f"Active learning init failed (non-fatal): {e}", exc_info=True)

    # ── Stage 2.5: Enhanced Drift Detection (Phase 1) ──
    if settings.drift.enabled:
        try:
            from backend.app.core.drift_detector import init_drift_detector
            init_drift_detector()
            logger.info("Enhanced drift detector initialized")
        except Exception as e:
            logger.error(f"Drift detector init failed (non-fatal): {e}", exc_info=True)

    # ── Stage 2.7: Fundus Gate v2 learned model (non-fatal) ──
    if settings.fundus_gate.enabled and settings.fundus_gate.version == "v2":
        try:
            import os

            from src.data.fundus_gate_learned import LearnedFundusGate
            gate_path = settings.fundus_gate.model_path
            weights = gate_path if os.path.isfile(gate_path) else None
            learned_gate = LearnedFundusGate(weights_path=weights)
            learned_gate.eval()
            gate.set_learned_gate(learned_gate)
            logger.info("Fundus gate v2 learned model loaded%s",
                        f" from {gate_path}" if weights else " (ImageNet weights)")
        except Exception as e:
            logger.warning(f"Fundus gate v2 learned model failed to load (non-fatal): {e}")
            gate.set_learned_gate(None)

    # ── Stage 3: Agent Orchestrator (existing) ──
    orchestrator = None
    try:
        from backend.app.routers.review import review_gate
        from src.agents.orchestrator import AgentOrchestrator
        from src.governance.audit import AuditTrail

        orchestrator = AgentOrchestrator(
            model_service=model_service,
            review_gate=review_gate,
            audit_trail=AuditTrail(),
        )
        agents.set_orchestrator(orchestrator)
        await orchestrator.start()
        logger.info("Agent orchestrator started")
    except Exception as e:
        logger.error(f"Agent startup failed (non-fatal): {e}", exc_info=True)

    yield

    # ── Shutdown (reverse order) ──
    if orchestrator:
        await orchestrator.stop()
    model_service.unload()

    if settings.telemetry.enabled:
        try:
            from backend.app.core.telemetry import shutdown_telemetry
            shutdown_telemetry()
        except Exception:
            pass

    if settings.database.enabled:
        try:
            from backend.app.core.db import dispose_engine
            await dispose_engine()
        except Exception:
            pass

    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-label retinal disease classification with clinical knowledge graph reasoning",
    lifespan=lifespan,
)

# ── Global exception handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )

# ── Middleware stack (order matters: last added = first executed) ──

# 1. CORS - allow credentials for SaaS frontend (cookies + Authorization header)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. GZIP compression for large responses (explainability payloads, base64 images)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. Request ID tracing
from backend.app.middleware.request_id import RequestIDMiddleware

app.add_middleware(RequestIDMiddleware)

# 4. Rate limiting
from backend.app.middleware.rate_limit import RateLimiterMiddleware

app.add_middleware(RateLimiterMiddleware, requests_per_minute=settings.rate_limit_per_minute)

# 5. Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=(self)"
    if settings.environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# ── Routers ──
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(auth.router)
app.include_router(review.router)
app.include_router(clinical.router)
app.include_router(explain.router)
app.include_router(analytics.router)
app.include_router(agents.router)
app.include_router(governance.router)
app.include_router(gate.router)
app.include_router(predict_edge.router)
app.include_router(offline.router)
app.include_router(quantized.router)
app.include_router(monitoring.router)
app.include_router(orgs.router)
app.include_router(billing.router)

# Phase 2: Voice-first
from backend.app.routers import voice

app.include_router(voice.router)

# Phase 3: Uganda health ecosystem
from backend.app.routers import dhis2, dicom, fhir, payments, sms

app.include_router(dhis2.router)
app.include_router(payments.router)
app.include_router(sms.router)
app.include_router(fhir.router)
app.include_router(dicom.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

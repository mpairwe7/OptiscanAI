"""
OpenTelemetry instrumentation for RetinalAI Clinical Screening Platform.

Provides distributed tracing, custom metrics, and log correlation for:
- FastAPI HTTP request auto-instrumentation
- Model inference spans with clinical attributes
- LLM call spans (Claude/Groq) with latency tracking
- Explainability method spans
- Custom metric instruments for prediction counts, drift scores, etc.

All instrumentation is opt-in via TELEMETRY__ENABLED=true. When disabled,
get_tracer() and get_meter() return no-op instances with zero overhead.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Module-level state ──────────────────────────────────────────────────────

_tracer_provider = None
_meter_provider = None
_tracer = None
_meter = None
_instruments: dict[str, Any] = {}
_initialized = False


# ── Initialization & Shutdown ───────────────────────────────────────────────


def init_telemetry(app: Any = None) -> None:
    """Initialize OpenTelemetry tracing, metrics, and FastAPI auto-instrumentation.

    Called once during FastAPI lifespan startup. No-op when telemetry is disabled.

    Parameters
    ----------
    app : FastAPI, optional
        The FastAPI application instance for auto-instrumentation.
    """
    global _tracer_provider, _meter_provider, _tracer, _meter, _initialized

    if not settings.telemetry.enabled:
        logger.info("OpenTelemetry disabled (TELEMETRY__ENABLED=false)")
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.semconv.resource import ResourceAttributes
    except ImportError:
        logger.warning(
            "OpenTelemetry SDK not installed. "
            "Run: pip install 'retinal-disease-mlops[observability]'"
        )
        return

    cfg = settings.telemetry

    # Resource identifies this service in traces/metrics
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: cfg.service_name,
            ResourceAttributes.SERVICE_VERSION: settings.app_version,
            "deployment.environment": settings.environment,
            "deployment.region": settings.deployment_region,
        }
    )

    # ── Tracing ──
    try:
        if cfg.otlp_protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

        span_exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint)
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(_tracer_provider)
        _tracer = trace.get_tracer("retinalai", settings.app_version)
    except Exception as e:
        logger.error(f"Failed to initialize OTEL tracing: {e}")
        _tracer = trace.get_tracer("retinalai")  # no-op fallback

    # ── Metrics ──
    try:
        if cfg.otlp_protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )

        metric_exporter = OTLPMetricExporter(endpoint=cfg.otlp_endpoint)
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=cfg.metrics_export_interval_ms,
        )
        _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(_meter_provider)
        _meter = metrics.get_meter("retinalai", settings.app_version)
    except Exception as e:
        logger.error(f"Failed to initialize OTEL metrics: {e}")
        _meter = metrics.get_meter("retinalai")  # no-op fallback

    # ── Create metric instruments ──
    _create_instruments()

    # ── FastAPI auto-instrumentation ──
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI auto-instrumentation enabled")
        except ImportError:
            logger.warning(
                "FastAPI OTEL instrumentation not available. "
                "Install: opentelemetry-instrumentation-fastapi"
            )
        except Exception as e:
            logger.error(f"FastAPI auto-instrumentation failed: {e}")

    _initialized = True
    logger.info(
        f"OpenTelemetry initialized: service={cfg.service_name}, "
        f"endpoint={cfg.otlp_endpoint}, sample_rate={cfg.sample_rate}"
    )


def shutdown_telemetry() -> None:
    """Flush pending spans/metrics and shutdown providers. Call in lifespan shutdown."""
    global _tracer_provider, _meter_provider, _initialized

    if not _initialized:
        return

    try:
        if _tracer_provider and hasattr(_tracer_provider, "shutdown"):
            _tracer_provider.shutdown()
        if _meter_provider and hasattr(_meter_provider, "shutdown"):
            _meter_provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")
    except Exception as e:
        logger.error(f"OpenTelemetry shutdown error: {e}")
    finally:
        _initialized = False


# ── Tracer & Meter Access ───────────────────────────────────────────────────


def get_tracer():
    """Return the configured tracer, or a no-op tracer if OTEL is disabled."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace

        return trace.get_tracer("retinalai")
    except ImportError:
        return _NoOpTracer()


def get_meter():
    """Return the configured meter, or a no-op meter if OTEL is disabled."""
    global _meter
    if _meter is not None:
        return _meter
    try:
        from opentelemetry import metrics

        return metrics.get_meter("retinalai")
    except ImportError:
        return _NoOpMeter()


# ── Custom Metric Instruments ───────────────────────────────────────────────


def _create_instruments() -> None:
    """Create all custom metric instruments."""
    global _instruments
    m = get_meter()
    _instruments = {
        "prediction_count": m.create_counter(
            name="retinalai.prediction.count",
            description="Total predictions served",
            unit="1",
        ),
        "inference_duration": m.create_histogram(
            name="retinalai.inference.duration_ms",
            description="Model inference latency in milliseconds",
            unit="ms",
        ),
        "model_confidence": m.create_histogram(
            name="retinalai.model.max_confidence",
            description="Maximum prediction confidence per request",
            unit="1",
        ),
        "diseases_detected": m.create_histogram(
            name="retinalai.prediction.diseases_detected",
            description="Number of diseases detected per prediction",
            unit="1",
        ),
        "active_learning_queue": m.create_up_down_counter(
            name="retinalai.active_learning.queue_size",
            description="Active learning queue size",
            unit="1",
        ),
        "drift_check_count": m.create_counter(
            name="retinalai.drift.check_count",
            description="Number of drift checks performed",
            unit="1",
        ),
        "review_count": m.create_counter(
            name="retinalai.review.count",
            description="Number of human reviews completed",
            unit="1",
        ),
        "llm_call_duration": m.create_histogram(
            name="retinalai.llm.call_duration_ms",
            description="LLM call latency in milliseconds",
            unit="ms",
        ),
    }


def record_prediction_metrics(
    inference_ms: float,
    diseases_detected: int,
    max_confidence: float,
    referral_priority: str,
    model_version: str = "default",
) -> None:
    """Record prediction-related metrics. Safe to call when OTEL is disabled."""
    if not _instruments:
        return

    attrs = {"model_version": model_version, "referral_priority": referral_priority}

    try:
        _instruments["prediction_count"].add(1, attrs)
        _instruments["inference_duration"].record(inference_ms, {"model_version": model_version})
        _instruments["model_confidence"].record(max_confidence, {"model_version": model_version})
        _instruments["diseases_detected"].record(
            diseases_detected, {"model_version": model_version}
        )
    except Exception:
        pass  # metrics are best-effort


def record_llm_metrics(
    provider: str,
    duration_ms: float,
    success: bool,
) -> None:
    """Record LLM call metrics."""
    if not _instruments:
        return
    try:
        _instruments["llm_call_duration"].record(
            duration_ms,
            {"provider": provider, "success": str(success)},
        )
    except Exception:
        pass


def record_review_metric(decision: str) -> None:
    """Record a human review event."""
    if not _instruments:
        return
    try:
        _instruments["review_count"].add(1, {"decision": decision})
    except Exception:
        pass


def record_drift_check() -> None:
    """Record that a drift check was performed."""
    if not _instruments:
        return
    try:
        _instruments["drift_check_count"].add(1)
    except Exception:
        pass


# ── Request ID → Span Bridge ───────────────────────────────────────────────


def inject_request_id(request_id: str) -> None:
    """Set request_id as a span attribute on the current active span.

    Called from RequestIDMiddleware to bridge the existing X-Request-ID
    tracing system into OpenTelemetry span context.
    """
    if not _initialized:
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("request.id", request_id)
    except Exception:
        pass


# ── @traced Decorator ───────────────────────────────────────────────────────


def traced(
    span_name: str,
    attributes: dict[str, str | int | float] | None = None,
) -> Callable[[F], F]:
    """Decorator to wrap a sync or async function in a custom OTEL span.

    Usage::

        @traced("model.inference", {"model": "vignn"})
        def predict(self, image):
            ...

        @traced("llm.claude")
        async def invoke_claude(prompt):
            ...
    """

    def decorator(fn: F) -> F:
        import asyncio

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    try:
                        result = await fn(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        raise

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    try:
                        result = fn(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ── No-Op Fallbacks ─────────────────────────────────────────────────────────


class _NoOpSpan:
    """Minimal no-op span for when OTEL is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def is_recording(self) -> bool:
        return False


class _NoOpTracer:
    """Minimal no-op tracer for when OTEL is not installed."""

    def start_as_current_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpMeter:
    """Minimal no-op meter for when OTEL is not installed."""

    def create_counter(self, **kwargs) -> "_NoOpInstrument":
        return _NoOpInstrument()

    def create_histogram(self, **kwargs) -> "_NoOpInstrument":
        return _NoOpInstrument()

    def create_up_down_counter(self, **kwargs) -> "_NoOpInstrument":
        return _NoOpInstrument()

    def create_observable_gauge(self, **kwargs) -> "_NoOpInstrument":
        return _NoOpInstrument()


class _NoOpInstrument:
    """No-op instrument that silently discards all recordings."""

    def add(self, amount: float = 1, attributes: dict | None = None) -> None:
        pass

    def record(self, amount: float = 0, attributes: dict | None = None) -> None:
        pass

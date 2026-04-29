"""Adds unique request ID to each request for tracing."""
import uuid
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Phase 1: Bridge request ID into OpenTelemetry span context
        try:
            from backend.app.core.telemetry import inject_request_id
            inject_request_id(request_id)
        except Exception:
            pass

        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            f"{request.method} {request.url.path} {response.status_code} {elapsed_ms:.1f}ms",
            extra={"request_id": request_id, "latency_ms": elapsed_ms, "endpoint": request.url.path, "status_code": response.status_code},
        )
        return response

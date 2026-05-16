"""In-memory sliding-window rate limiter (ASGI middleware)."""

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter per client IP.

    Supports X-Forwarded-For for clients behind reverse proxies.
    Returns 429 with Retry-After header when limit is exceeded.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self, key: str) -> None:
        now = time.time()
        self._buckets[key] = [t for t in self._buckets[key] if now - t < 60]

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        self._cleanup(client_ip)

        if len(self._buckets[client_ip]) >= self.rpm:
            logger.warning(
                "Rate limit exceeded",
                extra={"client_ip": client_ip, "rpm": self.rpm},
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )

        self._buckets[client_ip].append(time.time())
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.rpm - len(self._buckets[client_ip]))
        )
        return response

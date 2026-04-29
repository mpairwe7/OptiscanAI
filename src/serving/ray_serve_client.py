"""
Async HTTP client for calling Ray Serve deployments from FastAPI.

Provides ``RayServeClient`` which wraps ``httpx.AsyncClient`` with:
    - Circuit breaker protection (via ``src.serving.circuit_breaker``)
    - Configurable timeouts from ``backend.app.core.config.settings.ray``
    - Health-check probing
    - Structured logging

Usage inside a FastAPI lifespan::

    from src.serving.ray_serve_client import RayServeClient

    client = RayServeClient()

    @app.post("/predict")
    async def predict(file: UploadFile):
        result = await client.predict(await file.read())
        return result

    @app.on_event("shutdown")
    async def shutdown():
        await client.close()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_ray_settings() -> Dict[str, Any]:
    """Load Ray Serve client settings from the central config.

    Returns a plain dict so the module works even when the config system
    is not fully initialised (e.g. during unit tests).
    """
    try:
        from backend.app.core.config import settings

        return {
            "serve_url": settings.ray.serve_url,
            "timeout_s": settings.ray.timeout_s,
        }
    except Exception:
        logger.debug(
            "Could not load RayServeSettings; using defaults "
            "(serve_url=http://localhost:8000, timeout_s=30.0)"
        )
        return {
            "serve_url": "http://localhost:8000",
            "timeout_s": 30.0,
        }


def _load_circuit_breaker_settings() -> Dict[str, Any]:
    """Load circuit breaker thresholds from the central config."""
    try:
        from backend.app.core.config import settings

        cb = settings.circuit_breaker
        return {
            "failure_threshold": cb.failure_threshold,
            "recovery_timeout_s": cb.recovery_timeout_s,
            "half_open_max_calls": cb.half_open_max_calls,
        }
    except Exception:
        return {
            "failure_threshold": 5,
            "recovery_timeout_s": 60.0,
            "half_open_max_calls": 3,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RayServeClient:
    """Async HTTP client for calling a Ray Serve deployment.

    The underlying ``httpx.AsyncClient`` is created lazily on the first
    call so instantiation is cheap and does not require an event loop.

    Parameters
    ----------
    serve_url : str or None
        Base URL of the Ray Serve HTTP proxy (e.g. ``http://localhost:8000``).
        When *None*, the value is read from ``settings.ray.serve_url``.
    timeout_s : float or None
        Per-request timeout in seconds.  Defaults to ``settings.ray.timeout_s``.
    predict_route : str
        Route suffix appended to *serve_url* for prediction requests.
    health_route : str
        Route suffix for health probing.  Ray Serve exposes ``/-/healthz``
        on the HTTP proxy by default.
    """

    def __init__(
        self,
        serve_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        predict_route: str = "/predict",
        health_route: str = "/-/healthz",
    ) -> None:
        cfg = _load_ray_settings()

        self._serve_url: str = (serve_url or cfg["serve_url"]).rstrip("/")
        self._timeout_s: float = timeout_s if timeout_s is not None else cfg["timeout_s"]
        self._predict_route: str = predict_route
        self._health_route: str = health_route

        # Lazy-initialised httpx client
        self._client: Optional[Any] = None  # httpx.AsyncClient

        # Circuit breaker (lazy-initialised)
        self._circuit_breaker: Optional[Any] = None  # CircuitBreaker

        logger.info(
            "RayServeClient configured",
            extra={
                "serve_url": self._serve_url,
                "timeout_s": self._timeout_s,
                "predict_route": self._predict_route,
            },
        )

    # -- lazy init --------------------------------------------------------

    def _get_http_client(self) -> Any:
        """Return the httpx.AsyncClient, creating it on first use."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._serve_url,
                timeout=httpx.Timeout(
                    timeout=self._timeout_s,
                    connect=min(self._timeout_s, 10.0),
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
                headers={"Content-Type": "application/octet-stream"},
            )
            logger.debug("httpx.AsyncClient created for %s", self._serve_url)
        return self._client

    def _get_circuit_breaker(self) -> Any:
        """Return the CircuitBreaker for the Ray Serve backend."""
        if self._circuit_breaker is None:
            from src.serving.circuit_breaker import CircuitBreaker

            cb_cfg = _load_circuit_breaker_settings()
            self._circuit_breaker = CircuitBreaker(
                name="ray_serve",
                failure_threshold=cb_cfg["failure_threshold"],
                recovery_timeout_s=cb_cfg["recovery_timeout_s"],
                half_open_max_calls=cb_cfg["half_open_max_calls"],
            )
            logger.debug(
                "CircuitBreaker created for ray_serve (threshold=%d)",
                cb_cfg["failure_threshold"],
            )
        return self._circuit_breaker

    # -- public API -------------------------------------------------------

    async def predict(
        self,
        image_bytes: bytes,
        model_version: str = "default",
    ) -> Dict[str, Any]:
        """Send an image to Ray Serve for inference.

        The call is wrapped with a circuit breaker; if the breaker is
        open a ``CircuitBreakerError`` propagates to the caller so
        FastAPI can return an appropriate 503.

        Parameters
        ----------
        image_bytes : bytes
            Raw JPEG/PNG bytes of the fundus image.
        model_version : str
            Deployment version tag (passed as a query param so Ray Serve
            can route to canary vs. primary).

        Returns
        -------
        dict
            Parsed JSON response from the Ray Serve deployment containing
            ``predictions``, ``top_5``, ``num_classes``, etc.

        Raises
        ------
        src.serving.circuit_breaker.CircuitBreakerError
            When the circuit breaker is open and no fallback is provided.
        httpx.HTTPStatusError
            On non-2xx responses from Ray Serve.
        httpx.TimeoutException
            When the request exceeds ``timeout_s``.
        """
        cb = self._get_circuit_breaker()

        async def _do_predict() -> Dict[str, Any]:
            client = self._get_http_client()
            start = time.perf_counter()

            url = self._predict_route
            params: Dict[str, str] = {}
            if model_version != "default":
                params["version"] = model_version

            response = await client.post(
                url,
                content=image_bytes,
                params=params if params else None,
            )
            response.raise_for_status()

            latency_ms = (time.perf_counter() - start) * 1000
            result: Dict[str, Any] = response.json()

            result["client_metadata"] = {
                "round_trip_ms": round(latency_ms, 2),
                "model_version": model_version,
                "serve_url": self._serve_url,
            }

            logger.info(
                "Ray Serve prediction succeeded",
                extra={
                    "latency_ms": round(latency_ms, 2),
                    "model_version": model_version,
                    "status_code": response.status_code,
                },
            )
            return result

        # Execute through the circuit breaker
        return await cb.call(_do_predict)

    async def health_check(self) -> bool:
        """Probe the Ray Serve HTTP proxy health endpoint.

        Returns
        -------
        bool
            ``True`` if the proxy responds with HTTP 2xx, ``False`` otherwise.
        """
        try:
            client = self._get_http_client()
            response = await client.get(self._health_route)

            healthy = response.status_code < 300
            logger.debug(
                "Ray Serve health check: %s (status=%d)",
                "healthy" if healthy else "unhealthy",
                response.status_code,
            )
            return healthy

        except Exception as exc:
            logger.warning(
                "Ray Serve health check failed: %s",
                exc,
                extra={"serve_url": self._serve_url},
            )
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources.

        Safe to call multiple times or if the client was never opened.
        """
        if self._client is not None:
            try:
                await self._client.aclose()
                logger.info("RayServeClient HTTP connection closed")
            except Exception as exc:
                logger.warning("Error closing httpx client: %s", exc)
            finally:
                self._client = None

    # -- context manager support ------------------------------------------

    async def __aenter__(self) -> "RayServeClient":
        """Support ``async with RayServeClient() as client:``."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # -- repr -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RayServeClient("
            f"serve_url={self._serve_url!r}, "
            f"timeout_s={self._timeout_s}, "
            f"predict_route={self._predict_route!r})"
        )

"""Graceful degradation with fallback chains for RetinalAI.

Phase 4 future-proofing module.  Provides automatic service-health
monitoring and tiered degradation so the platform can continue
operating when upstream dependencies (LLM agents, Ray Serve, Kafka,
MLflow, etc.) are unhealthy.

Degradation levels (highest fidelity to lowest):

* ``FULL``            -- all services healthy, full agentic pipeline
* ``AGENT_DEGRADED``  -- LLM agent unavailable, rules-based clinical
                         reasoning still active, model inference OK
* ``RULES_ONLY``      -- agents + some infrastructure down, deterministic
                         rules pipeline only
* ``MODEL_ONLY``      -- bare model inference, no clinical reasoning or
                         agents

The ``GracefulDegradationManager`` periodically health-checks
registered services and automatically determines the current level.
``HealthAwareRouter`` consumes the level to route prediction requests
through the appropriate pipeline depth.

Integrates with:
* ``src.serving.circuit_breaker`` for per-service circuit breakers
* ``src.agents.event_bus`` for ``GRACEFUL_DEGRADATION_ACTIVATED``
  events on level changes
* ``backend.app.core.config.settings.resilience`` for configuration
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Awaitable, Callable

# Project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Degradation levels
# ---------------------------------------------------------------------------


class DegradationLevel(IntEnum):
    """Operational degradation levels, ordered from highest fidelity to lowest.

    The numeric values allow direct comparison (lower is better):
    ``DegradationLevel.FULL < DegradationLevel.MODEL_ONLY``.
    """

    FULL = 0
    AGENT_DEGRADED = 1
    RULES_ONLY = 2
    MODEL_ONLY = 3


# ---------------------------------------------------------------------------
# Service health tracking
# ---------------------------------------------------------------------------


@dataclass
class ServiceHealth:
    """Health state for a single registered service.

    Attributes
    ----------
    name : str
        Service identifier (e.g. ``"claude_agent"``, ``"ray_serve"``).
    healthy : bool
        ``True`` if the most recent health check passed.
    last_check : float
        Monotonic timestamp of the last check (``time.monotonic()``).
    consecutive_failures : int
        Number of sequential failures.  Resets to 0 on success.
    latency_ms : float
        Wall-clock duration of the last health check in milliseconds.
    """

    name: str
    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Degradation manager
# ---------------------------------------------------------------------------


class GracefulDegradationManager:
    """Monitors service health and determines the operational degradation level.

    Usage::

        manager = GracefulDegradationManager()
        manager.register_service("claude_agent", check_claude_health)
        manager.register_service("ray_serve", check_ray_health)

        # Periodic health sweep (call from a background task)
        await manager.check_all()

        # Query current level
        level = manager.current_level
    """

    # Services whose failure triggers each degradation level.  The
    # manager infers the level from which categories of services are
    # currently unhealthy.
    _AGENT_SERVICES = {"claude_agent", "groq_agent"}
    _INFRA_SERVICES = {"ray_serve", "kafka", "mlflow"}

    def __init__(self) -> None:
        self._services: dict[str, ServiceHealth] = {}
        self._health_checks: dict[str, Callable[..., Awaitable[bool]]] = {}
        self._current_level = DegradationLevel.FULL
        self._level_changed_at: str | None = None
        self._check_count: int = 0

        cfg = settings.resilience
        self._enabled: bool = cfg.enabled
        self._health_check_interval_s: float = cfg.health_check_interval_s

        logger.info(
            "GracefulDegradationManager created (enabled=%s, interval=%.1fs)",
            self._enabled,
            self._health_check_interval_s,
        )

    # ------------------------------------------------------------------
    # Service registration
    # ------------------------------------------------------------------

    def register_service(
        self,
        name: str,
        health_check_fn: Callable[..., Awaitable[bool]],
    ) -> None:
        """Register an async health check for a named service.

        Parameters
        ----------
        name : str
            Unique service identifier (e.g. ``"claude_agent"``).
        health_check_fn : Callable[..., Awaitable[bool]]
            Async callable that returns ``True`` when the service is
            healthy, ``False`` or raises on failure.
        """
        self._services[name] = ServiceHealth(name=name)
        self._health_checks[name] = health_check_fn
        logger.info("Registered health check for service '%s'", name)

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def check_all(self) -> dict[str, ServiceHealth]:
        """Run all registered health checks concurrently.

        Returns
        -------
        dict[str, ServiceHealth]
            Updated health states keyed by service name.
        """
        self._check_count += 1
        tasks = {
            name: self._check_service(name, fn)
            for name, fn in self._health_checks.items()
        }

        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Re-evaluate degradation level after all checks complete
        new_level = self._compute_level()
        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            self._level_changed_at = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "Degradation level changed: %s -> %s",
                old_level.name,
                new_level.name,
            )
            await self._emit_level_change(old_level, new_level)

        return dict(self._services)

    async def _check_service(
        self,
        name: str,
        health_fn: Callable[..., Awaitable[bool]],
    ) -> None:
        """Execute a single service health check with timing."""
        svc = self._services[name]
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(health_fn(), timeout=10.0)
            healthy = bool(result)
        except asyncio.TimeoutError:
            healthy = False
            logger.warning("Health check for '%s' timed out", name)
        except Exception as exc:
            healthy = False
            logger.warning("Health check for '%s' failed: %s", name, exc)

        elapsed_ms = (time.monotonic() - t0) * 1000
        svc.last_check = time.monotonic()
        svc.latency_ms = elapsed_ms

        if healthy:
            svc.healthy = True
            svc.consecutive_failures = 0
        else:
            svc.healthy = False
            svc.consecutive_failures += 1

    # ------------------------------------------------------------------
    # Level computation
    # ------------------------------------------------------------------

    def _compute_level(self) -> DegradationLevel:
        """Determine the degradation level from current service health.

        Logic:
        * If all services healthy -> FULL
        * If only agent services down -> AGENT_DEGRADED
        * If agent + some infra down -> RULES_ONLY
        * If the core model service is also down -> MODEL_ONLY
        """
        if not self._services:
            return DegradationLevel.FULL

        unhealthy = {
            name for name, svc in self._services.items() if not svc.healthy
        }

        if not unhealthy:
            return DegradationLevel.FULL

        agents_down = bool(unhealthy & self._AGENT_SERVICES)
        infra_down = bool(unhealthy & self._INFRA_SERVICES)
        model_down = "model_service" in unhealthy

        if model_down:
            return DegradationLevel.MODEL_ONLY

        if agents_down and infra_down:
            return DegradationLevel.RULES_ONLY

        if agents_down:
            return DegradationLevel.AGENT_DEGRADED

        # Infra-only degradation: agents still work, treat as partial
        if infra_down:
            return DegradationLevel.AGENT_DEGRADED

        return DegradationLevel.FULL

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> DegradationLevel:
        """The current operational degradation level."""
        return self._current_level

    # ------------------------------------------------------------------
    # Fallback execution
    # ------------------------------------------------------------------

    async def execute_with_fallback(
        self,
        primary: Callable[..., Awaitable[Any]],
        fallbacks: list[Callable[..., Awaitable[Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Try the primary callable, then each fallback in order.

        Parameters
        ----------
        primary : Callable
            The preferred async operation.
        fallbacks : list[Callable]
            Ordered list of fallback async operations to try on failure.
        *args, **kwargs
            Arguments forwarded to each callable.

        Returns
        -------
        dict
            ``{result, level, fallback_used, error}``
            ``fallback_used`` is ``None`` when the primary succeeded,
            otherwise the 0-based index of the fallback that succeeded.
            ``error`` is ``None`` on success.
        """
        # Try primary
        try:
            result = await primary(*args, **kwargs)
            return {
                "result": result,
                "level": self._current_level.name,
                "fallback_used": None,
                "error": None,
            }
        except Exception as primary_error:
            logger.warning(
                "Primary operation failed (%s), trying fallbacks",
                primary_error,
            )

        # Try fallbacks in order
        last_error: Exception | None = None
        for idx, fallback_fn in enumerate(fallbacks):
            try:
                result = await fallback_fn(*args, **kwargs)
                logger.info("Fallback #%d succeeded", idx)
                return {
                    "result": result,
                    "level": self._current_level.name,
                    "fallback_used": idx,
                    "error": None,
                }
            except Exception as fb_error:
                last_error = fb_error
                logger.warning("Fallback #%d failed: %s", idx, fb_error)

        # All fallbacks exhausted
        error_msg = str(last_error) if last_error else str(primary_error)
        logger.error("All fallbacks exhausted: %s", error_msg)
        return {
            "result": None,
            "level": self._current_level.name,
            "fallback_used": None,
            "error": error_msg,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a monitoring-friendly status summary.

        Returns
        -------
        dict
            Includes the current level, per-service health, total
            checks, and timestamp of the last level change.
        """
        return {
            "enabled": self._enabled,
            "current_level": self._current_level.name,
            "level_value": int(self._current_level),
            "level_changed_at": self._level_changed_at,
            "total_checks": self._check_count,
            "services": {
                name: {
                    "healthy": svc.healthy,
                    "consecutive_failures": svc.consecutive_failures,
                    "latency_ms": round(svc.latency_ms, 2),
                    "last_check_ago_s": round(
                        time.monotonic() - svc.last_check, 1
                    ) if svc.last_check > 0 else None,
                }
                for name, svc in self._services.items()
            },
        }

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def _emit_level_change(
        self,
        old_level: DegradationLevel,
        new_level: DegradationLevel,
    ) -> None:
        """Emit a ``GRACEFUL_DEGRADATION_ACTIVATED`` event on the event bus."""
        try:
            from src.agents.event_bus import event_bus, Event, EventType

            await event_bus.emit(Event(
                type=EventType.GRACEFUL_DEGRADATION_ACTIVATED,
                source="graceful_degradation_manager",
                data={
                    "old_level": old_level.name,
                    "new_level": new_level.name,
                    "services": {
                        name: svc.healthy
                        for name, svc in self._services.items()
                    },
                },
            ))
            logger.info("GRACEFUL_DEGRADATION_ACTIVATED event emitted")
        except Exception as exc:
            logger.debug(
                "Event bus unavailable for degradation event: %s", exc
            )


# ---------------------------------------------------------------------------
# Health-aware router
# ---------------------------------------------------------------------------


class HealthAwareRouter:
    """Routes prediction requests through the appropriate pipeline depth
    based on the current degradation level.

    Integrates with:
    * ``GracefulDegradationManager`` for level awareness
    * ``CircuitBreakerRegistry`` for per-service protection
    * ``ModelService`` for bare model inference

    Usage::

        router = HealthAwareRouter(degradation_manager)
        result = await router.route_prediction(image, threshold=0.5)
    """

    def __init__(
        self,
        degradation_manager: GracefulDegradationManager,
    ) -> None:
        self._manager = degradation_manager
        self._predictions_routed: int = 0
        self._fallback_predictions: int = 0

        logger.info("HealthAwareRouter created")

    async def route_prediction(
        self,
        image: Any,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Route a prediction through the appropriate pipeline.

        Parameters
        ----------
        image : PIL.Image.Image or similar
            The retinal image to classify.
        threshold : float | None
            Confidence threshold.  Passed through to the model service.

        Returns
        -------
        dict
            Prediction result augmented with routing metadata:
            ``{..., routing: {level, pipeline, fallback_used}}``.
        """
        level = self._manager.current_level
        self._predictions_routed += 1

        pipeline: str
        result: dict[str, Any]

        if level == DegradationLevel.FULL:
            pipeline = "full_agentic"
            result = await self._full_pipeline(image, threshold)

        elif level == DegradationLevel.AGENT_DEGRADED:
            pipeline = "rules_plus_model"
            result = await self._rules_pipeline(image, threshold)

        elif level == DegradationLevel.RULES_ONLY:
            pipeline = "rules_only"
            result = await self._rules_pipeline(image, threshold)

        else:
            pipeline = "model_only"
            result = await self._model_only_pipeline(image, threshold)
            self._fallback_predictions += 1

        result["routing"] = {
            "level": level.name,
            "pipeline": pipeline,
            "fallback_used": level > DegradationLevel.FULL,
        }

        return result

    # ------------------------------------------------------------------
    # Pipeline implementations
    # ------------------------------------------------------------------

    async def _full_pipeline(
        self,
        image: Any,
        threshold: float | None,
    ) -> dict[str, Any]:
        """Full pipeline: model inference + clinical reasoning + agents.

        Delegates to the model service and then attempts agent-based
        clinical narrative generation.
        """
        base_result = await self._model_only_pipeline(image, threshold)

        # Attempt agent enhancement via circuit breaker
        try:
            from src.serving.circuit_breaker import get_circuit_breaker_registry

            registry = get_circuit_breaker_registry()
            claude_cb = registry.get_or_create(
                "claude_agent",
                failure_threshold=settings.circuit_breaker.failure_threshold,
                recovery_timeout_s=settings.circuit_breaker.recovery_timeout_s,
            )

            async def _agent_enhance() -> dict[str, Any]:
                """Invoke the LangGraph screening agent for clinical narrative."""
                try:
                    from src.agents.graph import run_screening

                    report = await run_screening(
                        image=image,
                        scan_id=f"degrade-{int(time.time())}",
                        threshold=threshold,
                    )
                    return {
                        "agent_narrative": report.get("report", {}).get(
                            "clinical_narrative", "Agent analysis complete."
                        ),
                        "agent_available": True,
                        "scan_id": report.get("scan_id"),
                    }
                except ImportError:
                    return {
                        "agent_narrative": "Agent module not available.",
                        "agent_available": False,
                    }

            agent_result = await claude_cb.call(_agent_enhance)
            base_result["agent"] = agent_result

        except Exception as exc:
            logger.debug("Agent enhancement skipped: %s", exc)
            base_result["agent"] = {
                "agent_narrative": None,
                "agent_available": False,
            }

        return base_result

    async def _rules_pipeline(
        self,
        image: Any,
        threshold: float | None,
    ) -> dict[str, Any]:
        """Rules pipeline: model inference + deterministic clinical reasoning.

        Uses the knowledge graph for clinical reasoning without LLM
        agents.
        """
        base_result = await self._model_only_pipeline(image, threshold)
        base_result["agent"] = {
            "agent_narrative": None,
            "agent_available": False,
            "reason": "agent_degraded",
        }
        return base_result

    async def _model_only_pipeline(
        self,
        image: Any,
        threshold: float | None,
    ) -> dict[str, Any]:
        """Bare model inference with no clinical reasoning overlay."""
        try:
            from backend.app.core.model_service import model_service

            if not model_service.is_loaded:
                return {
                    "error": "Model not loaded",
                    "predictions": [],
                    "total_detected": 0,
                    "model_loaded": False,
                }

            # model_service.predict is synchronous; run in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                model_service.predict,
                image,
                threshold,
            )
            return result

        except Exception as exc:
            logger.error("Model-only pipeline failed: %s", exc)
            return {
                "error": str(exc),
                "predictions": [],
                "total_detected": 0,
                "model_loaded": False,
            }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_routing_info(self) -> dict[str, Any]:
        """Return routing statistics and current degradation state.

        Returns
        -------
        dict
            Includes prediction counts, fallback counts, and the
            current degradation status from the manager.
        """
        return {
            "predictions_routed": self._predictions_routed,
            "fallback_predictions": self._fallback_predictions,
            "current_level": self._manager.current_level.name,
            "degradation_status": self._manager.get_status(),
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_manager: GracefulDegradationManager | None = None
_router: HealthAwareRouter | None = None


def get_degradation_manager() -> GracefulDegradationManager | None:
    """Return the global degradation manager, or ``None`` if not initialised."""
    return _manager


def get_health_router() -> HealthAwareRouter | None:
    """Return the global health-aware router, or ``None`` if not initialised."""
    return _router


def init_graceful_degradation() -> None:
    """Create the module-level singletons.

    Respects ``settings.resilience.enabled``.  Registers default health
    checks for known services (agent, circuit breaker registry).
    """
    global _manager, _router

    if not settings.resilience.enabled:
        logger.info(
            "Graceful degradation disabled (RESILIENCE__ENABLED=false)"
        )
        return

    _manager = GracefulDegradationManager()

    # Register default health checks for known services
    async def _check_model_service() -> bool:
        try:
            from backend.app.core.model_service import model_service
            return model_service.is_loaded
        except Exception:
            return False

    async def _check_claude_agent() -> bool:
        """Check Claude agent availability via circuit breaker state."""
        try:
            from src.serving.circuit_breaker import get_circuit_breaker_registry
            registry = get_circuit_breaker_registry()
            cb = registry.get("claude_agent")
            if cb is None:
                return True  # no breaker registered means not yet tested
            return cb.state.value != "open"
        except Exception:
            return False

    async def _check_groq_agent() -> bool:
        """Check Groq fallback agent availability."""
        try:
            from src.serving.circuit_breaker import get_circuit_breaker_registry
            registry = get_circuit_breaker_registry()
            cb = registry.get("groq_agent")
            if cb is None:
                return True
            return cb.state.value != "open"
        except Exception:
            return False

    _manager.register_service("model_service", _check_model_service)
    _manager.register_service("claude_agent", _check_claude_agent)
    _manager.register_service("groq_agent", _check_groq_agent)

    _router = HealthAwareRouter(degradation_manager=_manager)

    logger.info(
        "Graceful degradation initialised (%d services registered)",
        len(_manager._services),
    )

"""
Circuit breaker pattern for external service calls.

Protects against cascading failures when calling LLMs (Claude/Groq),
Ray Serve, Kafka, or MLflow. Three states:
    CLOSED -> calls pass through normally
    OPEN   -> calls are immediately rejected (fallback used)
    HALF_OPEN -> limited calls allowed to test recovery

Integrates with Phase 1 OpenTelemetry for observability and
Phase 1 event bus for CIRCUIT_BREAKER_OPENED/CLOSED events.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and a call is rejected."""

    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is OPEN — call rejected")


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_transitions: int = 0
    consecutive_failures: int = 0


class CircuitBreaker:
    """Circuit breaker for async service calls with automatic fallback.

    Parameters
    ----------
    name : str
        Identifier for this breaker (e.g., "claude", "groq", "ray_serve").
    failure_threshold : int
        Number of consecutive failures before opening the circuit.
    recovery_timeout_s : float
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    half_open_max_calls : int
        Maximum concurrent test calls allowed in HALF_OPEN state.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, considering automatic OPEN -> HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout_s:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: Callable[..., Awaitable[T]] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with circuit breaker protection.

        Parameters
        ----------
        fn : Callable
            The async function to execute.
        fallback : Callable, optional
            Async fallback function to call when the circuit is open.
        *args, **kwargs
            Arguments forwarded to fn (and fallback if used).

        Returns
        -------
        The result of fn or fallback.

        Raises
        ------
        CircuitBreakerError
            If the circuit is open and no fallback is provided.
        """
        current_state = self.state

        self._stats.total_calls += 1

        if current_state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            logger.warning(f"Circuit breaker '{self.name}' is OPEN — rejecting call")

            if fallback is not None:
                return await fallback(*args, **kwargs)
            raise CircuitBreakerError(self.name)

        if current_state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    if fallback is not None:
                        return await fallback(*args, **kwargs)
                    raise CircuitBreakerError(self.name)
                self._half_open_calls += 1

        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            if fallback is not None:
                logger.info(
                    f"Circuit breaker '{self.name}' — using fallback after error: {e}"
                )
                return await fallback(*args, **kwargs)
            raise

    async def _on_success(self) -> None:
        """Record a successful call."""
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Recovery confirmed — close the circuit
            async with self._lock:
                self._half_open_calls = 0
            await self._transition(CircuitState.CLOSED)
            logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")

        self._consecutive_failures = 0
        self._stats.consecutive_failures = 0

    async def _on_failure(self, error: Exception) -> None:
        """Record a failed call and potentially open the circuit."""
        self._consecutive_failures += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures = self._consecutive_failures
        self._last_failure_time = time.monotonic()
        self._stats.last_failure_time = self._last_failure_time

        logger.warning(
            f"Circuit breaker '{self.name}' failure "
            f"({self._consecutive_failures}/{self.failure_threshold}): {error}"
        )

        if self._consecutive_failures >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                await self._transition(CircuitState.OPEN)
                logger.error(
                    f"Circuit breaker '{self.name}' OPENED after "
                    f"{self._consecutive_failures} consecutive failures"
                )

    async def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state and emit event."""
        old_state = self._state
        self._state = new_state
        self._stats.state_transitions += 1

        if new_state == CircuitState.OPEN:
            self._half_open_calls = 0

        # Emit event bus notification
        try:
            from src.agents.event_bus import Event, EventType, event_bus

            if new_state == CircuitState.OPEN:
                event_type = EventType.CIRCUIT_BREAKER_OPENED
            elif new_state == CircuitState.CLOSED and old_state in (
                CircuitState.OPEN,
                CircuitState.HALF_OPEN,
            ):
                event_type = EventType.CIRCUIT_BREAKER_CLOSED
            else:
                return

            await event_bus.emit(Event(
                type=event_type,
                source=f"circuit_breaker.{self.name}",
                data={
                    "breaker": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "consecutive_failures": self._consecutive_failures,
                },
            ))
        except Exception:
            pass  # event emission is best-effort

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_status(self) -> dict:
        """Return current breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout_s,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful": self._stats.successful_calls,
                "failed": self._stats.failed_calls,
                "rejected": self._stats.rejected_calls,
                "state_transitions": self._stats.state_transitions,
            },
        }


class CircuitBreakerRegistry:
    """Registry of named circuit breakers for all external services.

    Usage::

        registry = CircuitBreakerRegistry()
        claude_cb = registry.get_or_create("claude", failure_threshold=3)
        result = await claude_cb.call(invoke_claude, prompt, fallback=invoke_groq)
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> CircuitBreaker:
        """Get an existing breaker or create a new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_s=recovery_timeout_s,
                half_open_max_calls=half_open_max_calls,
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a breaker by name, or None if it doesn't exist."""
        return self._breakers.get(name)

    def get_all_status(self) -> list[dict]:
        """Return status of all registered breakers."""
        return [cb.get_status() for cb in self._breakers.values()]

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED state."""
        for cb in self._breakers.values():
            cb.reset()


# Module-level singleton
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry

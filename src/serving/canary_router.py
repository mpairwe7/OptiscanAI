"""
Canary routing for model version traffic splitting.

Provides deterministic, sticky-session-aware routing between a primary
model version and a canary candidate.  Used by the API layer and Ray
Serve to gradually roll out new model versions while monitoring for
regressions.

Features:
    - Consistent hashing (CRC-32) for sticky sessions -- the same
      ``request_id`` always routes to the same version.
    - Dynamic weight adjustment without restart.
    - Per-version call counters and real-time traffic-split metrics.
    - Settings-driven defaults via ``backend.app.core.config.settings.canary``.

Usage::

    router = CanaryRouter.from_settings()
    version = router.route(request_id="patient-abc-123")
    # version is either settings.canary.primary_version or canary_version

    router.update_weights(canary_weight=0.2)  # shift 20 % to canary
"""

from __future__ import annotations

import logging
import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------


@dataclass
class CanaryConfig:
    """Immutable snapshot of canary routing parameters.

    Attributes
    ----------
    primary_version : str
        Identifier of the primary (stable) model version.
    canary_version : str
        Identifier of the canary (experimental) model version.
        Empty string means canary is disabled.
    canary_weight : float
        Fraction of traffic routed to the canary, in ``[0.0, 1.0]``.
        ``0.0`` sends everything to primary; ``1.0`` sends everything
        to canary.
    sticky_sessions : bool
        When ``True``, the same ``request_id`` always maps to the same
        version (consistent hashing).  When ``False``, routing is purely
        probabilistic.
    """

    primary_version: str = "default"
    canary_version: str = ""
    canary_weight: float = 0.0
    sticky_sessions: bool = True

    def __post_init__(self) -> None:
        self.canary_weight = max(0.0, min(1.0, self.canary_weight))


# -----------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------


class CanaryRouter:
    """Deterministic canary router with per-version metrics.

    Parameters
    ----------
    config : CanaryConfig
        Initial routing configuration.
    """

    def __init__(self, config: Optional[CanaryConfig] = None) -> None:
        self._config = config or CanaryConfig()
        self._lock = threading.Lock()

        # Per-version counters
        self._call_counts: Dict[str, int] = {
            self._config.primary_version: 0,
        }
        if self._config.canary_version:
            self._call_counts[self._config.canary_version] = 0

        self._total_calls: int = 0

        logger.info(
            "CanaryRouter initialized: primary=%s, canary=%s, weight=%.2f, sticky=%s",
            self._config.primary_version,
            self._config.canary_version or "(none)",
            self._config.canary_weight,
            self._config.sticky_sessions,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> CanaryRouter:
        """Create a ``CanaryRouter`` from application settings.

        Reads ``settings.canary`` (a ``CanarySettings`` instance).

        Returns
        -------
        CanaryRouter
        """
        from backend.app.core.config import settings

        config = CanaryConfig(
            primary_version=settings.canary.primary_version,
            canary_version=settings.canary.canary_version,
            canary_weight=settings.canary.canary_weight,
            sticky_sessions=settings.canary.sticky_sessions,
        )
        return cls(config=config)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, request_id: str) -> str:
        """Decide which model version should serve this request.

        Parameters
        ----------
        request_id : str
            Unique identifier for the incoming request.  When
            ``sticky_sessions`` is enabled, the same ``request_id``
            always produces the same routing decision.

        Returns
        -------
        str
            The model version identifier (primary or canary).
        """
        with self._lock:
            config = self._config
            self._total_calls += 1

            # No canary configured or weight is zero -- fast path
            if not config.canary_version or config.canary_weight <= 0.0:
                self._increment(config.primary_version)
                return config.primary_version

            # All traffic to canary
            if config.canary_weight >= 1.0:
                self._increment(config.canary_version)
                return config.canary_version

            # Determine bucket
            if config.sticky_sessions:
                # CRC-32 gives a deterministic 32-bit integer for any string
                hash_value = zlib.crc32(request_id.encode("utf-8")) & 0xFFFFFFFF
                # Normalise to [0.0, 1.0)
                bucket = hash_value / 0x100000000
            else:
                import random
                bucket = random.random()

            if bucket < config.canary_weight:
                chosen = config.canary_version
            else:
                chosen = config.primary_version

            self._increment(chosen)

            logger.debug(
                "Routed request %s -> %s (bucket=%.4f, canary_weight=%.2f)",
                request_id,
                chosen,
                bucket,
                config.canary_weight,
            )
            return chosen

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, canary_weight: float) -> None:
        """Adjust the traffic split without restarting the router.

        Parameters
        ----------
        canary_weight : float
            New fraction of traffic for the canary version, ``[0.0, 1.0]``.
        """
        clamped = max(0.0, min(1.0, canary_weight))

        with self._lock:
            old_weight = self._config.canary_weight
            self._config = CanaryConfig(
                primary_version=self._config.primary_version,
                canary_version=self._config.canary_version,
                canary_weight=clamped,
                sticky_sessions=self._config.sticky_sessions,
            )
            # Ensure canary version counter exists
            if self._config.canary_version and self._config.canary_version not in self._call_counts:
                self._call_counts[self._config.canary_version] = 0

        logger.info(
            "Canary weight updated: %.2f -> %.2f (canary=%s)",
            old_weight,
            clamped,
            self._config.canary_version or "(none)",
        )

    def update_config(
        self,
        primary_version: Optional[str] = None,
        canary_version: Optional[str] = None,
        canary_weight: Optional[float] = None,
        sticky_sessions: Optional[bool] = None,
    ) -> None:
        """Update one or more routing parameters atomically.

        Parameters
        ----------
        primary_version : str, optional
        canary_version : str, optional
        canary_weight : float, optional
        sticky_sessions : bool, optional
        """
        with self._lock:
            new_primary = primary_version if primary_version is not None else self._config.primary_version
            new_canary = canary_version if canary_version is not None else self._config.canary_version
            new_weight = canary_weight if canary_weight is not None else self._config.canary_weight
            new_sticky = sticky_sessions if sticky_sessions is not None else self._config.sticky_sessions

            self._config = CanaryConfig(
                primary_version=new_primary,
                canary_version=new_canary,
                canary_weight=new_weight,
                sticky_sessions=new_sticky,
            )

            # Reset counters for new versions
            if new_primary not in self._call_counts:
                self._call_counts[new_primary] = 0
            if new_canary and new_canary not in self._call_counts:
                self._call_counts[new_canary] = 0

        logger.info(
            "CanaryRouter config updated: primary=%s, canary=%s, weight=%.2f, sticky=%s",
            new_primary,
            new_canary or "(none)",
            self._config.canary_weight,
            new_sticky,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, object]:
        """Return per-version call counts and traffic split metrics.

        Returns
        -------
        dict
            Keys: ``primary_version``, ``canary_version``,
            ``canary_weight``, ``sticky_sessions``, ``total_calls``,
            ``per_version_counts``, ``actual_traffic_split``.
        """
        with self._lock:
            config = self._config
            counts = dict(self._call_counts)
            total = self._total_calls

        # Compute actual traffic split
        actual_split: Dict[str, float] = {}
        if total > 0:
            for version, count in counts.items():
                actual_split[version] = round(count / total, 4)

        return {
            "primary_version": config.primary_version,
            "canary_version": config.canary_version or None,
            "canary_weight": config.canary_weight,
            "sticky_sessions": config.sticky_sessions,
            "total_calls": total,
            "per_version_counts": counts,
            "actual_traffic_split": actual_split,
        }

    def reset_metrics(self) -> None:
        """Zero all call counters."""
        with self._lock:
            for key in self._call_counts:
                self._call_counts[key] = 0
            self._total_calls = 0
        logger.info("CanaryRouter metrics reset")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _increment(self, version: str) -> None:
        """Increment the call counter for a version (caller holds lock)."""
        if version not in self._call_counts:
            self._call_counts[version] = 0
        self._call_counts[version] += 1

    def __repr__(self) -> str:
        return (
            f"CanaryRouter(primary={self._config.primary_version!r}, "
            f"canary={self._config.canary_version!r}, "
            f"weight={self._config.canary_weight:.2f}, "
            f"total_calls={self._total_calls})"
        )

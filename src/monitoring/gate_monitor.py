"""Fundus gate v2 monitoring — tracks pass/reject rates, disagreements, latency."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GateMetrics:
    """Snapshot of gate performance metrics."""

    total_checked: int = 0
    passed: int = 0
    rejected: int = 0
    pass_rate: float = 1.0
    rejection_by_layer: dict[str, int] = field(default_factory=dict)
    learned_statistical_disagreements: int = 0
    disagreement_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    alert_active: bool = False
    alert_message: str = ""


class GateMonitor:
    """Monitors fundus gate v2 decisions for operational alerting.

    Thread-safe for concurrent FastAPI usage. Uses deques with bounded
    size to prevent memory growth.
    """

    def __init__(
        self,
        alert_rejection_threshold: float = 0.15,
        alert_window_seconds: int = 3600,
        max_history: int = 5000,
    ):
        self._alert_threshold = alert_rejection_threshold
        self._alert_window = alert_window_seconds
        self._latencies: deque[float] = deque(maxlen=max_history)
        self._results: deque[tuple[float, bool, str]] = deque(maxlen=max_history)
        self._disagreements: int = 0
        self._total: int = 0
        self._passed: int = 0
        self._rejected: int = 0
        self._rejection_by_layer: dict[str, int] = {}
        self._start_time = time.time()

    def record(
        self,
        passed: bool,
        layer: str,
        latency_ms: float,
        statistical_passed: bool,
        learned_passed: Optional[bool],
    ) -> None:
        """Record a gate evaluation result."""
        now = time.time()
        self._total += 1
        self._latencies.append(latency_ms)
        self._results.append((now, passed, layer))

        if passed:
            self._passed += 1
        else:
            self._rejected += 1
            self._rejection_by_layer[layer] = self._rejection_by_layer.get(layer, 0) + 1

        # Track disagreements between statistical and learned gates
        if learned_passed is not None and statistical_passed != learned_passed:
            self._disagreements += 1

    def check_alert(self) -> Optional[dict]:
        """Check if rejection rate exceeds threshold in the alert window.

        Returns alert dict if triggered, None otherwise.
        """
        now = time.time()
        cutoff = now - self._alert_window

        recent = [(ts, passed, layer) for ts, passed, layer in self._results if ts > cutoff]
        if len(recent) < 10:
            return None  # Not enough data

        recent_rejections = sum(1 for _, passed, _ in recent if not passed)
        rejection_rate = recent_rejections / len(recent)

        if rejection_rate > self._alert_threshold:
            return {
                "alert": "high_rejection_rate",
                "rejection_rate": round(rejection_rate, 3),
                "threshold": self._alert_threshold,
                "window_seconds": self._alert_window,
                "recent_total": len(recent),
                "recent_rejections": recent_rejections,
                "message": (
                    f"Gate rejection rate {rejection_rate:.1%} exceeds "
                    f"threshold {self._alert_threshold:.1%} in the last "
                    f"{self._alert_window}s ({recent_rejections}/{len(recent)} rejected)"
                ),
            }
        return None

    def metrics(self) -> GateMetrics:
        """Compute current metrics snapshot."""
        latencies = list(self._latencies)
        if latencies:
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
        else:
            p50 = p95 = p99 = 0.0

        pass_rate = self._passed / max(self._total, 1)
        disagreement_rate = self._disagreements / max(self._total, 1)

        alert = self.check_alert()

        return GateMetrics(
            total_checked=self._total,
            passed=self._passed,
            rejected=self._rejected,
            pass_rate=round(pass_rate, 4),
            rejection_by_layer=dict(self._rejection_by_layer),
            learned_statistical_disagreements=self._disagreements,
            disagreement_rate=round(disagreement_rate, 4),
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            alert_active=alert is not None,
            alert_message=alert["message"] if alert else "",
        )

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time


# Global singleton
gate_monitor = GateMonitor()

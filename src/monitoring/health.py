"""
Production health monitoring - latency tracking, error rates, SLA compliance.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HealthReport:
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    total_predictions: int = 0
    total_errors: int = 0
    uptime_seconds: float = 0.0
    sla_compliant: bool = True


class HealthMonitor:
    """Tracks inference latency, errors, and throughput."""

    def __init__(self, max_latency_p99_ms: float = 100.0, max_error_rate: float = 0.05):
        self.latencies: deque[float] = deque(maxlen=5000)
        self.errors = 0
        self.total = 0
        self.start_time = time.time()
        self.max_latency = max_latency_p99_ms
        self.max_errors = max_error_rate

    def record(self, latency_ms: float, success: bool = True):
        self.latencies.append(latency_ms)
        self.total += 1
        if not success:
            self.errors += 1

    def report(self) -> HealthReport:
        if not self.latencies:
            return HealthReport(uptime_seconds=time.time() - self.start_time)

        lat = np.array(self.latencies)
        elapsed = time.time() - self.start_time
        error_rate = self.errors / max(self.total, 1)

        p99 = float(np.percentile(lat, 99))
        compliant = p99 <= self.max_latency and error_rate <= self.max_errors

        return HealthReport(
            latency_p50_ms=float(np.median(lat)),
            latency_p95_ms=float(np.percentile(lat, 95)),
            latency_p99_ms=p99,
            throughput_rps=self.total / max(elapsed, 1),
            error_rate=error_rate,
            total_predictions=self.total,
            total_errors=self.errors,
            uptime_seconds=elapsed,
            sla_compliant=compliant,
        )

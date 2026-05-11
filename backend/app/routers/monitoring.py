"""Monitoring, metrics, and governance endpoints for offline & mobile features.

Provides Prometheus-compatible metrics, admin dashboards, and governance
audit endpoints for offline RAG, mobile bundles, voice-first features,
and quantization performance tracking.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["monitoring"])


# ---------------------------------------------------------------------------
# Metrics storage (in-memory, exported to Prometheus)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe metrics collector for offline & mobile features.

    Tracks counters, gauges, and histograms for:
    - Offline mode usage (sessions, queries, fallbacks)
    - Offline faithfulness scores
    - Mobile bundle sizes and download counts
    - Voice-first latency and usage
    - Quantization performance (memory, latency, throughput)
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = {
            "offline_mode_sessions_total": 0,
            "offline_queries_total": 0,
            "offline_fallback_to_online_total": 0,
            "online_fallback_to_offline_total": 0,
            "mobile_bundle_downloads_total": 0,
            "delta_sync_total": 0,
            "delta_sync_failures_total": 0,
            "voice_sessions_total": 0,
            "voice_barge_in_total": 0,
            "voice_barge_in_success_total": 0,
            "quantized_inference_total": 0,
        }
        self._gauges: dict[str, float] = {
            "offline_faithfulness_score": 0.0,
            "mobile_bundle_size_mb": 0.0,
            "mobile_bundle_version": 0.0,
            "voice_first_latency_seconds_p95": 0.0,
            "voice_first_latency_seconds_p50": 0.0,
            "offline_rag_search_latency_seconds_p95": 0.0,
            "quantized_model_memory_mb": 0.0,
            "quantized_model_memory_reduction_pct": 0.0,
            "quantized_inference_latency_p95_ms": 0.0,
            "active_offline_users": 0,
            "active_voice_users": 0,
        }
        self._histograms: dict[str, list[float]] = {
            "offline_search_latency_ms": [],
            "voice_latency_ms": [],
            "delta_sync_duration_ms": [],
            "bundle_download_duration_ms": [],
        }

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        if name in self._counters:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        if name in self._gauges:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Add an observation to a histogram metric."""
        if name in self._histograms:
            self._histograms[name].append(value)
            # Keep only last 10000 observations
            if len(self._histograms[name]) > 10000:
                self._histograms[name] = self._histograms[name][-5000:]

    def get_all(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "mean": sum(v) / len(v) if v else 0.0,
                    "p50": sorted(v)[len(v) // 2] if v else 0.0,
                    "p95": sorted(v)[int(len(v) * 0.95)] if v else 0.0,
                    "p99": sorted(v)[int(len(v) * 0.99)] if v else 0.0,
                }
                for k, v in self._histograms.items()
            },
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: list[str] = []

        for name, value in self._counters.items():
            lines.append(f"# TYPE retinalai_{name} counter")
            lines.append(f"retinalai_{name} {value}")

        for name, value in self._gauges.items():
            lines.append(f"# TYPE retinalai_{name} gauge")
            lines.append(f"retinalai_{name} {value}")

        for name, observations in self._histograms.items():
            if not observations:
                continue
            sorted_obs = sorted(observations)
            n = len(sorted_obs)
            lines.append(f"# TYPE retinalai_{name} summary")
            lines.append(f'retinalai_{name}{{quantile="0.5"}} {sorted_obs[n // 2]:.4f}')
            lines.append(f'retinalai_{name}{{quantile="0.95"}} {sorted_obs[int(n * 0.95)]:.4f}')
            lines.append(f'retinalai_{name}{{quantile="0.99"}} {sorted_obs[int(n * 0.99)]:.4f}')
            lines.append(f"retinalai_{name}_count {n}")
            lines.append(f"retinalai_{name}_sum {sum(sorted_obs):.4f}")

        return "\n".join(lines) + "\n"


# Singleton instance
metrics_collector = MetricsCollector()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OfflineStats(BaseModel):
    """Aggregated statistics for offline mode usage."""

    total_offline_sessions: int = 0
    total_offline_queries: int = 0
    total_fallback_to_online: int = 0
    total_fallback_to_offline: int = 0
    active_offline_users: int = 0
    offline_faithfulness_score: float = 0.0
    offline_search_latency_p95_ms: float = 0.0
    bundle_version: str = ""
    bundle_size_mb: float = 0.0
    total_bundle_downloads: int = 0
    total_delta_syncs: int = 0
    delta_sync_failures: int = 0
    delta_sync_latency_p95_ms: float = 0.0


class VoiceStats(BaseModel):
    """Aggregated statistics for voice-first features."""

    total_voice_sessions: int = 0
    active_voice_users: int = 0
    voice_latency_p50_ms: float = 0.0
    voice_latency_p95_ms: float = 0.0
    total_barge_ins: int = 0
    barge_in_success_rate: float = 0.0


class QuantizationStats(BaseModel):
    """Aggregated statistics for quantization performance."""

    total_quantized_inferences: int = 0
    quantized_model_memory_mb: float = 0.0
    memory_reduction_pct: float = 0.0
    inference_latency_p95_ms: float = 0.0


class FullAdminStats(BaseModel):
    """Complete admin statistics for offline, mobile, and voice features."""

    offline: OfflineStats = Field(default_factory=OfflineStats)
    voice: VoiceStats = Field(default_factory=VoiceStats)
    quantization: QuantizationStats = Field(default_factory=QuantizationStats)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    uptime_seconds: float = 0.0


class GrafanaDashboard(BaseModel):
    """Grafana dashboard configuration for offline & mobile metrics."""

    dashboard_url: str = ""
    panels: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/offline_stats", response_model=OfflineStats)
async def get_offline_stats() -> OfflineStats:
    """Get aggregated offline mode statistics.

    Returns session counts, faithfulness scores, bundle information,
    and sync metrics for monitoring and governance.
    """
    m = metrics_collector.get_all()
    counters = m["counters"]
    gauges = m["gauges"]
    histograms = m["histograms"]

    return OfflineStats(
        total_offline_sessions=int(counters.get("offline_mode_sessions_total", 0)),
        total_offline_queries=int(counters.get("offline_queries_total", 0)),
        total_fallback_to_online=int(counters.get("offline_fallback_to_online_total", 0)),
        total_fallback_to_offline=int(counters.get("online_fallback_to_offline_total", 0)),
        active_offline_users=int(gauges.get("active_offline_users", 0)),
        offline_faithfulness_score=gauges.get("offline_faithfulness_score", 0.0),
        offline_search_latency_p95_ms=histograms.get("offline_search_latency_ms", {}).get("p95", 0.0),
        bundle_size_mb=gauges.get("mobile_bundle_size_mb", 0.0),
        total_bundle_downloads=int(counters.get("mobile_bundle_downloads_total", 0)),
        total_delta_syncs=int(counters.get("delta_sync_total", 0)),
        delta_sync_failures=int(counters.get("delta_sync_failures_total", 0)),
        delta_sync_latency_p95_ms=histograms.get("delta_sync_duration_ms", {}).get("p95", 0.0),
    )


@router.get("/voice_stats", response_model=VoiceStats)
async def get_voice_stats() -> VoiceStats:
    """Get aggregated voice-first feature statistics."""
    m = metrics_collector.get_all()
    counters = m["counters"]
    gauges = m["gauges"]
    histograms = m["histograms"]

    total_barge_in = int(counters.get("voice_barge_in_total", 0))
    barge_in_success = int(counters.get("voice_barge_in_success_total", 0))

    return VoiceStats(
        total_voice_sessions=int(counters.get("voice_sessions_total", 0)),
        active_voice_users=int(gauges.get("active_voice_users", 0)),
        voice_latency_p50_ms=histograms.get("voice_latency_ms", {}).get("p50", 0.0),
        voice_latency_p95_ms=histograms.get("voice_latency_ms", {}).get("p95", 0.0),
        total_barge_ins=total_barge_in,
        barge_in_success_rate=barge_in_success / total_barge_in if total_barge_in > 0 else 0.0,
    )


@router.get("/quantization_stats", response_model=QuantizationStats)
async def get_quantization_stats() -> QuantizationStats:
    """Get quantization performance statistics."""
    m = metrics_collector.get_all()
    counters = m["counters"]
    gauges = m["gauges"]

    return QuantizationStats(
        total_quantized_inferences=int(counters.get("quantized_inference_total", 0)),
        quantized_model_memory_mb=gauges.get("quantized_model_memory_mb", 0.0),
        memory_reduction_pct=gauges.get("quantized_model_memory_reduction_pct", 0.0),
        inference_latency_p95_ms=gauges.get("quantized_inference_latency_p95_ms", 0.0),
    )


@router.get("/stats", response_model=FullAdminStats)
async def get_full_admin_stats() -> FullAdminStats:
    """Get complete admin statistics for all new features.

    Aggregates offline, voice, and quantization metrics along with
    feature flag status and server uptime.
    """
    offline = await get_offline_stats()
    voice = await get_voice_stats()
    quant = await get_quantization_stats()

    # Collect feature flag status
    flags = {
        "FLAG_QUANTIZATION": getattr(getattr(settings, "quantization", None), "enabled", False),
        "FLAG_OFFLINE_RAG": getattr(getattr(settings, "offline_rag", None), "enabled", False),
        "FLAG_VOICE_FIRST_MOBILE": getattr(getattr(settings, "voice_first", None), "enabled", False),
        "FLAG_MOBILE_BUNDLE": getattr(getattr(settings, "mobile_bundle", None), "enabled", False),
        "FLAG_SPECULATIVE_DECODING": getattr(getattr(settings, "quantization", None), "speculative_decoding_enabled", False),
        "FLAG_PREFIX_CACHE": getattr(getattr(settings, "quantization", None), "prefix_cache_enabled", False),
    }

    return FullAdminStats(
        offline=offline,
        voice=voice,
        quantization=quant,
        feature_flags=flags,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/metrics/prometheus")
async def prometheus_metrics() -> str:
    """Export all metrics in Prometheus text exposition format.

    Suitable for scraping by Prometheus server.
    Returns text/plain with metrics in standard format.
    """
    from starlette.responses import PlainTextResponse

    return PlainTextResponse(
        content=metrics_collector.to_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/grafana/dashboard", response_model=GrafanaDashboard)
async def get_grafana_dashboard_config() -> GrafanaDashboard:
    """Get recommended Grafana dashboard configuration.

    Returns panel definitions for the "Offline & Mobile Experience"
    dashboard covering all new metrics.
    """
    return GrafanaDashboard(
        dashboard_url="/grafana/d/retinalai-offline-mobile",
        panels=[
            {
                "title": "Offline Mode Sessions",
                "type": "timeseries",
                "metric": "retinalai_offline_mode_sessions_total",
                "description": "Total offline mode sessions over time",
            },
            {
                "title": "Offline Faithfulness Score",
                "type": "gauge",
                "metric": "retinalai_offline_faithfulness_score",
                "thresholds": {"green": 0.82, "yellow": 0.75, "red": 0.0},
                "description": "Current offline RAG faithfulness score",
            },
            {
                "title": "Offline Search Latency (p95)",
                "type": "timeseries",
                "metric": "retinalai_offline_search_latency_ms{quantile=\"0.95\"}",
                "description": "p95 latency for offline RAG searches",
            },
            {
                "title": "Mobile Bundle Size",
                "type": "stat",
                "metric": "retinalai_mobile_bundle_size_mb",
                "unit": "MB",
                "thresholds": {"green": 0, "yellow": 120, "red": 150},
                "description": "Current mobile bundle size in MB",
            },
            {
                "title": "Delta Sync Duration (p95)",
                "type": "timeseries",
                "metric": "retinalai_delta_sync_duration_ms{quantile=\"0.95\"}",
                "description": "p95 delta sync duration",
            },
            {
                "title": "Voice-First Latency (p95)",
                "type": "timeseries",
                "metric": "retinalai_voice_latency_ms{quantile=\"0.95\"}",
                "description": "p95 end-to-end voice interaction latency",
            },
            {
                "title": "Barge-In Success Rate",
                "type": "gauge",
                "metric": "retinalai_voice_barge_in_success_total / retinalai_voice_barge_in_total",
                "thresholds": {"green": 0.92, "yellow": 0.80, "red": 0.0},
                "description": "Voice barge-in success rate (target >= 92%)",
            },
            {
                "title": "Quantized Model Memory",
                "type": "stat",
                "metric": "retinalai_quantized_model_memory_mb",
                "unit": "MB",
                "description": "Memory used by quantized model",
            },
            {
                "title": "Memory Reduction %",
                "type": "gauge",
                "metric": "retinalai_quantized_model_memory_reduction_pct",
                "thresholds": {"green": 38, "yellow": 25, "red": 0},
                "description": "Memory reduction from quantization (target >= 38%)",
            },
            {
                "title": "Active Offline Users",
                "type": "stat",
                "metric": "retinalai_active_offline_users",
                "description": "Currently active offline users",
            },
            {
                "title": "Active Voice Users",
                "type": "stat",
                "metric": "retinalai_active_voice_users",
                "description": "Currently active voice users",
            },
            {
                "title": "Online/Offline Fallback Ratio",
                "type": "timeseries",
                "metric": "retinalai_offline_fallback_to_online_total",
                "description": "Fallback events between online and offline modes",
            },
        ],
    )

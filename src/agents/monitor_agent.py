"""MonitorAgent: Continuous production monitoring and retraining orchestrator.

Runs on a periodic tick (configurable, default 60s) and:
1. Checks prediction logs for volume anomalies
2. Evaluates drift metrics from recent predictions
3. Monitors SLA compliance (latency, error rate)
4. Evaluates retraining triggers and emits events when thresholds are crossed
5. Tracks confidence distribution shifts over time
"""
import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents.base import BaseAgent, ToolResult
from src.agents.event_bus import EventType
from src.monitoring.drift import DriftReport
from src.training.retraining import RetrainingTrigger, RetrainingDecision

logger = logging.getLogger(__name__)

# Default monitoring window
WINDOW_SIZE = 200
CONFIDENCE_DROP_THRESHOLD = 0.10
ANOMALY_VOLUME_FACTOR = 3.0  # flag if volume > 3x rolling average


class MonitorAgent(BaseAgent):
    """Autonomous monitoring agent for production model health.

    Tools:
        check_drift: Analyze recent predictions for distribution shift
        check_sla: Verify latency and error rate SLA compliance
        evaluate_retraining: Run retraining decision logic
        check_volume_anomaly: Detect unusual prediction volume patterns
        get_confidence_trend: Track prediction confidence over time
    """

    def __init__(
        self,
        prediction_log_dir: str = "logs/predictions",
        tick_interval: float = 60.0,
        retraining_trigger: Optional[RetrainingTrigger] = None,
        **kwargs,
    ):
        super().__init__(name="monitor_agent", **kwargs)
        self.log_dir = Path(prediction_log_dir)
        self._tick_interval = tick_interval
        self.retraining = retraining_trigger or RetrainingTrigger()
        self._confidence_window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._latency_window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._volume_history: deque[int] = deque(maxlen=30)
        self._last_log_position: dict[str, int] = {}

    def loop_interval_seconds(self) -> float:
        return self._tick_interval

    async def setup(self):
        self.register_tool("check_drift", self._check_drift)
        self.register_tool("check_sla", self._check_sla)
        self.register_tool("evaluate_retraining", self._evaluate_retraining)
        self.register_tool("check_volume_anomaly", self._check_volume_anomaly)
        self.register_tool("get_confidence_trend", self._get_confidence_trend)

        # React to completed scans
        self.subscribe(EventType.SCAN_ANALYZED, self._on_scan_analyzed)

    async def tick(self):
        """Periodic monitoring cycle."""
        # Ingest new predictions from logs
        new_entries = self._read_new_predictions()
        for entry in new_entries:
            self._confidence_window.append(entry.get("avg_confidence", 0.5))
            self._latency_window.append(entry.get("inference_ms", 0))

        if len(self._confidence_window) < 10:
            return  # not enough data yet

        # Check all monitors
        drift_result = await self.use_tool("check_drift")
        sla_result = await self.use_tool("check_sla")
        retrain_result = await self.use_tool("evaluate_retraining")

        # Emit heartbeat with status
        await self.emit(EventType.HEARTBEAT, {
            "agent": self.name,
            "predictions_tracked": len(self._confidence_window),
            "drift_detected": drift_result.data.get("drift_detected", False) if drift_result.success else False,
            "sla_compliant": sla_result.data.get("compliant", True) if sla_result.success else True,
            "retrain_needed": retrain_result.data.get("should_retrain", False) if retrain_result.success else False,
        })

    async def _on_scan_analyzed(self, event):
        """Track metrics from each analyzed scan."""
        self.retraining.record_new_data(count=1)

    # ── Tool implementations ──

    async def _check_drift(self) -> ToolResult:
        """Analyze confidence distribution for drift signals."""
        if len(self._confidence_window) < 20:
            return ToolResult(tool="check_drift", success=True, data={"drift_detected": False, "reason": "insufficient data"})

        window = list(self._confidence_window)
        first_half = np.array(window[:len(window)//2])
        second_half = np.array(window[len(window)//2:])

        mean_shift = abs(float(np.mean(second_half) - np.mean(first_half)))
        drift_detected = mean_shift > CONFIDENCE_DROP_THRESHOLD

        if drift_detected:
            drift_report = DriftReport(
                drift_detected=True,
                severity="warning" if mean_shift < 0.2 else "critical",
                details=f"Confidence mean shifted by {mean_shift:.4f}",
            )
            self.retraining.record_drift(drift_report)
            await self.emit(EventType.DRIFT_DETECTED, {
                "severity": drift_report.severity,
                "mean_shift": round(mean_shift, 4),
                "window_size": len(window),
            })

        return ToolResult(
            tool="check_drift",
            success=True,
            data={
                "drift_detected": drift_detected,
                "mean_shift": round(mean_shift, 4),
                "first_half_mean": round(float(np.mean(first_half)), 4),
                "second_half_mean": round(float(np.mean(second_half)), 4),
                "window_size": len(window),
            },
        )

    async def _check_sla(self) -> ToolResult:
        """Verify latency and error rate against SLA targets."""
        if not self._latency_window:
            return ToolResult(tool="check_sla", success=True, data={"compliant": True, "reason": "no data"})

        latencies = np.array(list(self._latency_window))
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        # SLA: p99 < 100ms
        compliant = p99 < 100.0

        if not compliant:
            await self.emit(EventType.SLA_VIOLATED, {
                "p99_ms": round(p99, 2),
                "threshold_ms": 100.0,
                "p95_ms": round(p95, 2),
            })

        return ToolResult(
            tool="check_sla",
            success=True,
            data={
                "compliant": compliant,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "threshold_ms": 100.0,
                "sample_count": len(latencies),
            },
        )

    async def _evaluate_retraining(self) -> ToolResult:
        """Run the retraining decision engine."""
        decision: RetrainingDecision = self.retraining.evaluate()

        if decision.should_retrain:
            await self.emit(EventType.RETRAIN_TRIGGERED, {
                "reason": decision.reason,
                "priority": decision.priority,
            })

        return ToolResult(
            tool="evaluate_retraining",
            success=True,
            data=decision.to_dict(),
        )

    async def _check_volume_anomaly(self) -> ToolResult:
        """Detect unusual prediction volume patterns."""
        if len(self._volume_history) < 3:
            return ToolResult(tool="check_volume_anomaly", success=True, data={"anomaly": False})

        avg = np.mean(list(self._volume_history)[:-1])
        latest = self._volume_history[-1] if self._volume_history else 0

        anomaly = avg > 0 and latest > avg * ANOMALY_VOLUME_FACTOR

        if anomaly:
            await self.emit(EventType.ANOMALY_DETECTED, {
                "type": "volume_spike",
                "latest": latest,
                "average": round(float(avg), 1),
                "factor": round(latest / avg, 1) if avg > 0 else 0,
            })

        return ToolResult(
            tool="check_volume_anomaly",
            success=True,
            data={
                "anomaly": anomaly,
                "latest_volume": latest,
                "rolling_average": round(float(avg), 1),
            },
        )

    async def _get_confidence_trend(self) -> ToolResult:
        """Return confidence trend data for the dashboard."""
        if not self._confidence_window:
            return ToolResult(tool="get_confidence_trend", success=True, data={"trend": []})

        window = list(self._confidence_window)
        # Bucket into groups of 10 for trend line
        bucket_size = max(1, len(window) // 10)
        trend = []
        for i in range(0, len(window), bucket_size):
            bucket = window[i:i+bucket_size]
            trend.append({
                "index": i,
                "mean_confidence": round(float(np.mean(bucket)), 4),
                "count": len(bucket),
            })

        return ToolResult(
            tool="get_confidence_trend",
            success=True,
            data={
                "trend": trend,
                "overall_mean": round(float(np.mean(window)), 4),
                "overall_std": round(float(np.std(window)), 4),
            },
        )

    # ── Helpers ──

    def _read_new_predictions(self) -> list[dict]:
        """Read new prediction log entries since last check."""
        entries = []
        if not self.log_dir.exists():
            return entries

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"predictions_{today}.jsonl"
        if not log_file.exists():
            return entries

        last_pos = self._last_log_position.get(str(log_file), 0)
        try:
            with open(log_file) as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            # Compute average confidence from top predictions
                            preds = entry.get("top_predictions", [])
                            if preds:
                                avg_conf = np.mean([p.get("probability", 0.5) for p in preds])
                                entry["avg_confidence"] = float(avg_conf)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
                self._last_log_position[str(log_file)] = f.tell()
        except Exception as e:
            logger.warning(f"Failed to read prediction log: {e}")

        return entries

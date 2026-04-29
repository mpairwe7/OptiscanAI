"""Automated retraining triggers based on drift, schedule, and data volume."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.monitoring.drift import DriftReport

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    should_retrain: bool
    reason: str
    priority: str = "normal"  # normal | high | critical
    triggered_at: str = ""

    def to_dict(self) -> dict:
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "priority": self.priority,
            "triggered_at": self.triggered_at,
        }


class RetrainingTrigger:
    """Evaluates conditions for automated model retraining."""

    def __init__(
        self,
        drift_threshold: str = "warning",
        max_days_since_training: int = 30,
        min_new_samples: int = 500,
        performance_drop_threshold: float = 0.05,
        state_file: str = "outputs/retraining_state.json",
    ):
        self.drift_threshold = drift_threshold
        self.max_days = max_days_since_training
        self.min_new_samples = min_new_samples
        self.perf_drop_threshold = performance_drop_threshold
        self.state_file = Path(state_file)
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {
            "last_training": datetime.now(timezone.utc).isoformat(),
            "last_metrics": {},
            "new_samples_count": 0,
            "drift_events": 0,
        }

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def record_new_data(self, count: int = 1):
        """Record new data samples arriving."""
        self._state["new_samples_count"] = self._state.get("new_samples_count", 0) + count
        self._save_state()

    def record_training(self, metrics: dict):
        """Record that a training run completed."""
        self._state["last_training"] = datetime.now(timezone.utc).isoformat()
        self._state["last_metrics"] = metrics
        self._state["new_samples_count"] = 0
        self._state["drift_events"] = 0
        self._save_state()

    def record_drift(self, drift_report: DriftReport):
        """Record a drift detection event."""
        if drift_report.drift_detected:
            self._state["drift_events"] = self._state.get("drift_events", 0) + 1
            self._save_state()

    def evaluate(
        self,
        current_metrics: Optional[dict] = None,
        drift_report: Optional[DriftReport] = None,
    ) -> RetrainingDecision:
        """Evaluate all retraining triggers and return decision."""
        now = datetime.now(timezone.utc)
        reasons = []
        priority = "normal"

        # Trigger 1: Time-based
        last_training = datetime.fromisoformat(
            self._state.get("last_training", now.isoformat())
        )
        days_elapsed = (now - last_training).days
        if days_elapsed > self.max_days:
            reasons.append(
                f"Model is {days_elapsed} days old (threshold: {self.max_days})"
            )
            priority = "high"

        # Trigger 2: Data volume
        new_samples = self._state.get("new_samples_count", 0)
        if new_samples >= self.min_new_samples:
            reasons.append(
                f"{new_samples} new samples available (threshold: {self.min_new_samples})"
            )

        # Trigger 3: Drift-based
        if drift_report and drift_report.drift_detected:
            if drift_report.severity == "critical":
                reasons.append(f"Critical drift detected: {drift_report.details}")
                priority = "critical"
            elif drift_report.severity == "warning":
                reasons.append(f"Warning-level drift: {drift_report.details}")

        drift_events = self._state.get("drift_events", 0)
        if drift_events >= 5:
            reasons.append(f"{drift_events} drift events accumulated")
            priority = max(
                priority,
                "high",
                key=lambda x: {"normal": 0, "high": 1, "critical": 2}[x],
            )

        # Trigger 4: Performance degradation
        if current_metrics and self._state.get("last_metrics"):
            last_f1 = self._state["last_metrics"].get("f1_macro", 0)
            current_f1 = current_metrics.get("f1_macro", 0)
            if last_f1 > 0 and (last_f1 - current_f1) > self.perf_drop_threshold:
                reasons.append(
                    f"F1 dropped from {last_f1:.4f} to {current_f1:.4f}"
                )
                priority = "critical"

        should_retrain = len(reasons) > 0
        decision = RetrainingDecision(
            should_retrain=should_retrain,
            reason="; ".join(reasons) if reasons else "No retraining needed",
            priority=priority,
            triggered_at=now.isoformat() if should_retrain else "",
        )

        if should_retrain:
            logger.warning(
                f"Retraining triggered ({priority}): {decision.reason}"
            )

        return decision

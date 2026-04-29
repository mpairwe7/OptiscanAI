"""Logs all predictions for audit trail and monitoring."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Append-only prediction log for audit and drift analysis."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir or settings.prediction_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._current_date = None

    def _get_log_file(self) -> Path:
        """Rotate log file daily."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._current_file = self.log_dir / f"predictions_{today}.jsonl"
        return self._current_file

    def log(
        self,
        request_id: str,
        user: str,
        predictions: list[dict],
        threshold: float | None,
        threshold_source: str,
        inference_ms: float,
        model_loaded: bool,
        image_size: tuple[int, int] = (0, 0),
        num_detected: int = 0,
        referral_priority: str = "",
        fundus_gate_version: str = "",
        learned_score: float = -1.0,
        statistical_score: float = -1.0,
        fusion_confidence: float = -1.0,
    ):
        """Log a single prediction event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "user": user,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "inference_ms": inference_ms,
            "model_loaded": model_loaded,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "num_detected": num_detected,
            "referral_priority": referral_priority,
            "top_predictions": predictions[:5],  # Top 5 only for storage efficiency
            "fundus_gate_version": fundus_gate_version,
            "fundus_gate_learned_score": learned_score,
            "fundus_gate_statistical_score": statistical_score,
            "fundus_gate_fusion_confidence": fusion_confidence,
        }
        try:
            log_file = self._get_log_file()
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def log_gate_rejection(
        self,
        request_id: str,
        user: str,
        gate_result,
        image_size: tuple[int, int] = (0, 0),
    ):
        """Log a gate rejection event for audit and model improvement."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "gate_rejection",
            "request_id": request_id,
            "user": user,
            "passed": gate_result.passed,
            "confidence": gate_result.confidence,
            "layer": gate_result.layer,
            "reason": gate_result.reason,
            "image_width": image_size[0],
            "image_height": image_size[1],
        }
        # V2 fields (check via hasattr for backward compat)
        if hasattr(gate_result, "statistical_confidence"):
            entry["statistical_confidence"] = gate_result.statistical_confidence
        if hasattr(gate_result, "learned_confidence"):
            entry["learned_confidence"] = gate_result.learned_confidence
        if hasattr(gate_result, "fused_confidence"):
            entry["fused_confidence"] = gate_result.fused_confidence
        if hasattr(gate_result, "latency_ms"):
            entry["gate_latency_ms"] = gate_result.latency_ms
        if hasattr(gate_result, "gate_version"):
            entry["gate_version"] = gate_result.gate_version
        try:
            log_file = self._get_log_file()
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log gate rejection: {e}")


# Global singleton
prediction_logger = PredictionLogger()

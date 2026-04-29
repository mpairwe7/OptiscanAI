"""Immutable audit trail for model lifecycle events.
Records training runs, deployments, predictions, and configuration changes."""
import json
import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    MODEL_TRAINED = "model_trained"
    MODEL_EVALUATED = "model_evaluated"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"
    DATA_VALIDATED = "data_validated"
    DRIFT_DETECTED = "drift_detected"
    CONFIG_CHANGED = "config_changed"
    PREDICTION_FLAGGED = "prediction_flagged"
    HUMAN_REVIEW = "human_review"
    FAIRNESS_EVALUATED = "fairness_evaluated"


@dataclass
class AuditEvent:
    event_type: str
    timestamp: str
    actor: str  # user or system component
    details: dict
    model_version: str = ""
    checksum: str = ""  # SHA-256 of event for integrity


class AuditTrail:
    """Append-only audit log with integrity verification."""

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._last_checksum = ""

    def _compute_checksum(self, event_data: str) -> str:
        """Chain checksums for tamper detection."""
        content = f"{self._last_checksum}{event_data}"
        return hashlib.sha256(content.encode()).hexdigest()

    def log_event(self, event_type: AuditEventType, actor: str, details: dict, model_version: str = ""):
        """Record an audit event."""
        event_data = json.dumps({"type": event_type.value, "actor": actor, "details": details, "model": model_version})
        checksum = self._compute_checksum(event_data)
        self._last_checksum = checksum

        event = AuditEvent(
            event_type=event_type.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            details=details,
            model_version=model_version,
            checksum=checksum,
        )

        # Append to daily log file
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

        logger.info(f"Audit: {event_type.value} by {actor} [{checksum[:8]}]")
        return event

    def log_training(self, actor: str, model_version: str, config: dict, metrics: dict):
        return self.log_event(AuditEventType.MODEL_TRAINED, actor, {"config": config, "metrics": metrics}, model_version)

    def log_evaluation(self, actor: str, model_version: str, metrics: dict):
        return self.log_event(AuditEventType.MODEL_EVALUATED, actor, {"metrics": metrics}, model_version)

    def log_deployment(self, actor: str, model_version: str, environment: str):
        return self.log_event(AuditEventType.MODEL_DEPLOYED, actor, {"environment": environment}, model_version)

    def log_drift(self, severity: str, details: dict, model_version: str = ""):
        return self.log_event(AuditEventType.DRIFT_DETECTED, "monitoring_system", {"severity": severity, **details}, model_version)

    def log_human_review(self, reviewer: str, prediction_id: str, decision: str, notes: str = ""):
        return self.log_event(AuditEventType.HUMAN_REVIEW, reviewer, {"prediction_id": prediction_id, "decision": decision, "notes": notes})

    def verify_integrity(self) -> bool:
        """Verify the chain of checksums hasn't been tampered with."""
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        prev_checksum = ""

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    event = json.loads(line.strip())
                    event_data = json.dumps({
                        "type": event["event_type"], "actor": event["actor"],
                        "details": event["details"], "model": event["model_version"],
                    })
                    expected = hashlib.sha256(f"{prev_checksum}{event_data}".encode()).hexdigest()
                    if event["checksum"] != expected:
                        logger.error(f"Integrity check FAILED at {event['timestamp']}")
                        return False
                    prev_checksum = event["checksum"]

        logger.info("Audit trail integrity verified")
        return True

    def get_events(self, event_type: AuditEventType = None, limit: int = 100) -> list[dict]:
        """Query recent audit events."""
        events = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event_type is None or event["event_type"] == event_type.value:
                        events.append(event)
                    if len(events) >= limit:
                        return events
        return events


# Global singleton
audit_trail = AuditTrail()

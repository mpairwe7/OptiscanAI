"""
ImmutableAuditLogger — Append-only audit logging for EU AI Act compliance.

Provides tamper-evident logging of all model predictions, reviews, and
lifecycle events. Each log entry includes a SHA-256 hash chain linking
it to the previous entry (blockchain-like integrity).

Storage backends:
    - Local JSON-lines file (default, for development)
    - Kafka topic (production, configure via KAFKA_BOOTSTRAP_SERVERS)
    - Apache Iceberg / WORM storage (enterprise, via plugin)

EU AI Act (Article 12): High-risk AI systems must maintain logs that
enable tracing of the AI system's operation throughout its lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    PREDICTION = "prediction"
    REVIEW = "human_review"
    MODEL_PROMOTION = "model_promotion"
    MODEL_ROLLBACK = "model_rollback"
    RETRAINING = "retraining_triggered"
    BIAS_AUDIT = "bias_audit"
    DATA_DRIFT = "data_drift_detected"
    EXPORT = "model_exported"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    model_version: str = ""
    user_id: Optional[str] = None
    patient_id_hash: Optional[str] = None  # Hashed for privacy
    payload: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry (excluding entry_hash)."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "patient_id_hash": self.patient_id_hash,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


class ImmutableAuditLogger:
    """Append-only audit logger with hash-chain integrity.

    Parameters
    ----------
    log_dir : str
        Directory for local log files.
    kafka_config : dict | None
        Kafka producer config (bootstrap_servers, topic, etc.).
        If provided, logs are also written to Kafka.
    max_file_size_mb : int
        Maximum log file size before rotation. Default 100MB.
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        kafka_config: Optional[Dict[str, Any]] = None,
        max_file_size_mb: int = 100,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.kafka_config = kafka_config
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Hash chain state (protected by _write_lock for thread safety)
        self._write_lock = threading.Lock()
        self._previous_hash = "genesis"
        self._entry_count = 0

        # Current log file
        self._current_file = self._get_log_file()

        # Kafka producer (lazy init)
        self._kafka_producer = None

        # Load chain state
        self._load_chain_state()

        logger.info(f"AuditLogger initialized: {self.log_dir} (chain length: {self._entry_count})")

    # ------------------------------------------------------------------
    # Core logging API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        model_version: str,
        predictions: Dict[str, float],
        uncertainty: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        image_hash: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> str:
        """Log a model prediction event."""
        payload = {
            "predictions": predictions,
            "uncertainty": uncertainty,
            "image_hash": image_hash,
            "latency_ms": latency_ms,
            "num_diseases_detected": sum(1 for v in predictions.values() if v >= 0.5),
        }

        patient_hash = self._hash_patient_id(patient_id) if patient_id else None

        return self._write_entry(
            event_type=AuditEventType.PREDICTION,
            model_version=model_version,
            patient_id_hash=patient_hash,
            payload=payload,
        )

    def log_review(
        self,
        model_version: str,
        reviewer_id: str,
        case_id: str,
        original_predictions: Dict[str, float],
        corrected_labels: Optional[Dict[str, float]] = None,
        approved: bool = False,
        notes: Optional[str] = None,
    ) -> str:
        """Log a human review event."""
        payload = {
            "case_id": case_id,
            "original_predictions": original_predictions,
            "corrected_labels": corrected_labels,
            "approved": approved,
            "notes": notes,
        }

        return self._write_entry(
            event_type=AuditEventType.REVIEW,
            model_version=model_version,
            user_id=reviewer_id,
            payload=payload,
        )

    def log_model_promotion(
        self,
        old_version: str,
        new_version: str,
        metrics: Dict[str, float],
        promoted_by: str,
        reason: str = "",
    ) -> str:
        """Log a model promotion (staging -> production)."""
        return self._write_entry(
            event_type=AuditEventType.MODEL_PROMOTION,
            model_version=new_version,
            user_id=promoted_by,
            payload={
                "old_version": old_version,
                "new_version": new_version,
                "metrics": metrics,
                "reason": reason,
            },
        )

    def log_bias_audit(
        self,
        model_version: str,
        passed: bool,
        violations: List[str],
        metrics_summary: Dict[str, Any],
    ) -> str:
        """Log a bias audit result."""
        return self._write_entry(
            event_type=AuditEventType.BIAS_AUDIT,
            model_version=model_version,
            payload={
                "passed": passed,
                "violations": violations,
                "metrics_summary": metrics_summary,
            },
        )

    def log_event(
        self,
        event_type: AuditEventType,
        model_version: str = "",
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Log a generic audit event."""
        return self._write_entry(
            event_type=event_type,
            model_version=model_version,
            user_id=user_id,
            payload=payload or {},
        )

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the hash chain of the entire audit log.

        Returns
        -------
        dict
            valid (bool), entries_checked (int), first_invalid_entry (str|None)
        """
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        previous_hash = "genesis"
        entries_checked = 0
        first_invalid = None

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry_data = json.loads(line)
                    entries_checked += 1

                    # Verify previous hash link
                    if entry_data.get("previous_hash") != previous_hash:
                        if first_invalid is None:
                            first_invalid = entry_data.get("event_id")

                    # Verify self-hash
                    entry = AuditEntry(**{
                        k: v for k, v in entry_data.items() if k != "entry_hash"
                    })
                    entry.previous_hash = entry_data.get("previous_hash", "")
                    computed = entry.compute_hash()

                    if computed != entry_data.get("entry_hash"):
                        if first_invalid is None:
                            first_invalid = entry_data.get("event_id")

                    previous_hash = entry_data.get("entry_hash", "")

        result = {
            "valid": first_invalid is None,
            "entries_checked": entries_checked,
            "first_invalid_entry": first_invalid,
        }

        logger.info(
            f"Chain verification: {'VALID' if result['valid'] else 'INVALID'} "
            f"({entries_checked} entries)"
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_entry(
        self,
        event_type: AuditEventType | str,
        model_version: str = "",
        user_id: Optional[str] = None,
        patient_id_hash: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write an audit entry to the log with hash chain linking.

        Thread-safe: acquires _write_lock to prevent hash chain forks from
        concurrent writes.
        """
        with self._write_lock:
            entry = AuditEntry(
                event_type=event_type.value if isinstance(event_type, AuditEventType) else event_type,
                model_version=model_version,
                user_id=user_id,
                patient_id_hash=patient_id_hash,
                payload=payload or {},
                previous_hash=self._previous_hash,
            )
            entry.entry_hash = entry.compute_hash()

            # Write to local file
            self._rotate_if_needed()
            with open(self._current_file, "a") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")

            # Update chain state
            self._previous_hash = entry.entry_hash
            self._entry_count += 1

        # Write to Kafka outside lock (non-critical, can be async)
        if self.kafka_config:
            self._send_to_kafka(entry)

        return entry.event_id

    def _get_log_file(self) -> Path:
        """Get current log file path."""
        existing = sorted(self.log_dir.glob("audit_*.jsonl"))
        if existing:
            latest = existing[-1]
            if latest.stat().st_size < self.max_file_size_bytes:
                return latest

        # Create new file
        idx = len(existing)
        return self.log_dir / f"audit_{idx:04d}.jsonl"

    def _rotate_if_needed(self):
        """Rotate log file if size exceeds limit."""
        if self._current_file.exists():
            if self._current_file.stat().st_size >= self.max_file_size_bytes:
                self._current_file = self._get_log_file()

    def _load_chain_state(self):
        """Load the last hash from existing logs to continue the chain."""
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        if not log_files:
            return

        # Read last entry from last file
        last_file = log_files[-1]
        last_line = None
        with open(last_file) as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if last_line:
            try:
                data = json.loads(last_line)
                self._previous_hash = data.get("entry_hash", "genesis")
                # Count all entries
                total = 0
                for lf in log_files:
                    with open(lf) as f:
                        total += sum(1 for line in f if line.strip())
                self._entry_count = total
            except json.JSONDecodeError:
                pass

    def _hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for privacy (HIPAA compliance)."""
        return hashlib.sha256(f"retinal-ai:{patient_id}".encode()).hexdigest()[:16]

    def _send_to_kafka(self, entry: AuditEntry):
        """Send entry to Kafka topic."""
        try:
            if self._kafka_producer is None:
                from kafka import KafkaProducer
                self._kafka_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
                    value_serializer=lambda v: json.dumps(v, default=str).encode(),
                )

            topic = self.kafka_config.get("topic", "retinal-ai-audit")
            self._kafka_producer.send(topic, value=asdict(entry))
        except Exception as e:
            logger.warning(f"Kafka send failed: {e}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_recent_entries(
        self, n: int = 100, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent audit entries."""
        entries = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            with open(log_file) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    if event_type and entry.get("event_type") != event_type:
                        continue
                    entries.append(entry)
                    if len(entries) >= n:
                        return entries
                except json.JSONDecodeError:
                    continue

        return entries

    def get_statistics(self) -> Dict[str, Any]:
        """Return audit log statistics."""
        log_files = list(self.log_dir.glob("audit_*.jsonl"))
        total_size = sum(f.stat().st_size for f in log_files)

        return {
            "total_entries": self._entry_count,
            "log_files": len(log_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "chain_head": self._previous_hash[:16] + "...",
        }

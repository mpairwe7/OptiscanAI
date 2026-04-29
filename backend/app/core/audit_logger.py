"""
ProductionAuditLogger -- Kafka-first audit logging with JSONL fallback.

Extends the governance-layer ``ImmutableAuditLogger`` pattern with:
    - Confluent-Kafka Producer for durable event streaming
    - Apache Iceberg sink awareness (via separate consumer)
    - Async interface for use in FastAPI request handlers
    - SHA-256 hash-chain integrity (EU AI Act Article 12 compliance)
    - Monthly JSONL rotation (``logs/audit/audit_YYYY-MM.jsonl``)

The logger is designed as a singleton; obtain it via ``get_audit_logger()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single immutable audit log entry with hash-chain linking."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    model_version: str = ""
    user_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry (excluding ``entry_hash``)."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


# -----------------------------------------------------------------------
# Production audit logger
# -----------------------------------------------------------------------


class ProductionAuditLogger:
    """Kafka-first, JSONL-fallback audit logger with SHA-256 chain.

    Initialization order:
        1. Try to create a confluent_kafka Producer (lazy import).
        2. If Kafka is unavailable, fall back to JSONL files.
        3. JSONL writes are always performed regardless of Kafka status
           to ensure a local audit trail exists.

    Parameters
    ----------
    log_dir : str
        Base directory for JSONL fallback files.
    max_file_size_mb : int
        Maximum JSONL file size before rotation (within a month).
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        max_file_size_mb: int = 100,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Hash-chain state (protected by lock)
        self._write_lock = threading.Lock()
        self._previous_hash: str = "genesis"
        self._entry_count: int = 0

        # Kafka producer (lazy)
        self._kafka_producer: Any = None
        self._kafka_available: bool = False
        self._kafka_topic: str = settings.kafka.audit_topic

        # Iceberg settings (consumed by external sink, stored for metadata)
        self._iceberg_enabled: bool = settings.iceberg.enabled
        self._iceberg_table: str = settings.iceberg.table_name

        # Initialise Kafka if enabled
        if settings.kafka.enabled:
            self._init_kafka()

        # Restore chain state from existing JSONL files
        self._load_chain_state()

        logger.info(
            "ProductionAuditLogger initialized: dir=%s, kafka=%s, "
            "iceberg_sink=%s, chain_length=%d",
            self.log_dir,
            self._kafka_available,
            self._iceberg_enabled,
            self._entry_count,
        )

    # ------------------------------------------------------------------
    # Kafka initialization
    # ------------------------------------------------------------------

    def _init_kafka(self) -> None:
        """Attempt to create a confluent_kafka Producer."""
        try:
            from confluent_kafka import Producer  # type: ignore[import-untyped]

            kafka_conf: Dict[str, Any] = {
                "bootstrap.servers": settings.kafka.bootstrap_servers,
                "acks": settings.kafka.acks,
                "security.protocol": settings.kafka.security_protocol,
                "client.id": "retinalai-audit-logger",
                "linger.ms": 50,
                "batch.num.messages": 100,
                "compression.type": "lz4",
            }
            self._kafka_producer = Producer(kafka_conf)
            self._kafka_available = True
            logger.info(
                "Kafka Producer created for topic '%s' at %s",
                self._kafka_topic,
                settings.kafka.bootstrap_servers,
            )
        except ImportError:
            logger.warning(
                "confluent_kafka not installed; using JSONL-only mode.  "
                "Install with: pip install confluent-kafka"
            )
            self._kafka_available = False
        except Exception as exc:
            logger.error("Kafka Producer init failed: %s — falling back to JSONL", exc)
            self._kafka_available = False

    # ------------------------------------------------------------------
    # Core logging API
    # ------------------------------------------------------------------

    async def log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        model_version: str = "",
        user_id: Optional[str] = None,
    ) -> str:
        """Create an audit entry and persist it.

        Writes to Kafka first (if available), then always to JSONL as a
        durable fallback.

        Parameters
        ----------
        event_type : str
            Category of the event (e.g. ``prediction``, ``model_promotion``).
        payload : dict
            Event-specific data.
        model_version : str, optional
            Model version string.
        user_id : str | None, optional
            Identifier of the user/service that triggered the event.

        Returns
        -------
        str
            The ``event_id`` of the created entry.
        """
        entry = self._create_entry(event_type, payload, model_version, user_id)

        # Kafka (best-effort, non-blocking)
        if self._kafka_available:
            self._send_to_kafka(entry)

        # JSONL (always, synchronous via thread-pool to avoid blocking the loop)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_jsonl, entry)

        return entry.event_id

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def query(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read recent audit entries from JSONL files.

        Parameters
        ----------
        event_type : str | None
            Filter by event type.  ``None`` returns all types.
        limit : int
            Maximum number of entries to return (most recent first).

        Returns
        -------
        list[dict]
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._query_sync, event_type, limit
        )

    def _query_sync(
        self, event_type: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Synchronous implementation of ``query``."""
        entries: List[Dict[str, Any]] = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            try:
                with open(log_file) as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()]
            except OSError as exc:
                logger.warning("Could not read %s: %s", log_file, exc)
                continue

            for line in reversed(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event_type and record.get("event_type") != event_type:
                    continue

                entries.append(record)
                if len(entries) >= limit:
                    return entries

        return entries

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the SHA-256 hash chain across all JSONL files.

        Returns
        -------
        dict
            ``valid`` (bool), ``entries_checked`` (int),
            ``first_invalid_entry`` (str | None), ``chain_head`` (str).
        """
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        previous_hash = "genesis"
        entries_checked = 0
        first_invalid: Optional[str] = None

        for log_file in log_files:
            try:
                with open(log_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        entries_checked += 1

                        # Check previous-hash link
                        if data.get("previous_hash") != previous_hash:
                            if first_invalid is None:
                                first_invalid = data.get("event_id", "unknown")

                        # Recompute self-hash
                        entry = AuditEntry(
                            event_id=data.get("event_id", ""),
                            event_type=data.get("event_type", ""),
                            timestamp=data.get("timestamp", 0.0),
                            model_version=data.get("model_version", ""),
                            user_id=data.get("user_id"),
                            payload=data.get("payload", {}),
                            previous_hash=data.get("previous_hash", ""),
                        )
                        computed = entry.compute_hash()
                        if computed != data.get("entry_hash"):
                            if first_invalid is None:
                                first_invalid = data.get("event_id", "unknown")

                        previous_hash = data.get("entry_hash", "")

            except OSError as exc:
                logger.warning("Could not read %s during verification: %s", log_file, exc)

        result = {
            "valid": first_invalid is None,
            "entries_checked": entries_checked,
            "first_invalid_entry": first_invalid,
            "chain_head": previous_hash[:16] + "..." if previous_hash != "genesis" else "genesis",
        }

        logger.info(
            "Chain verification: %s (%d entries)",
            "VALID" if result["valid"] else "INVALID",
            entries_checked,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_entry(
        self,
        event_type: str,
        payload: Dict[str, Any],
        model_version: str,
        user_id: Optional[str],
    ) -> AuditEntry:
        """Build an ``AuditEntry`` and update the hash chain."""
        with self._write_lock:
            entry = AuditEntry(
                event_type=event_type,
                model_version=model_version,
                user_id=user_id,
                payload=payload,
                previous_hash=self._previous_hash,
            )
            entry.entry_hash = entry.compute_hash()
            self._previous_hash = entry.entry_hash
            self._entry_count += 1
        return entry

    def _get_current_file(self) -> Path:
        """Return the JSONL file path for the current month.

        File naming: ``audit_YYYY-MM.jsonl`` with a numeric suffix if
        the file exceeds ``max_file_size_bytes``.
        """
        month_tag = datetime.now(timezone.utc).strftime("%Y-%m")
        base = self.log_dir / f"audit_{month_tag}.jsonl"

        if not base.exists() or base.stat().st_size < self.max_file_size_bytes:
            return base

        # Rotate with numeric suffix
        idx = 1
        while True:
            rotated = self.log_dir / f"audit_{month_tag}_{idx:03d}.jsonl"
            if not rotated.exists() or rotated.stat().st_size < self.max_file_size_bytes:
                return rotated
            idx += 1

    def _write_jsonl(self, entry: AuditEntry) -> None:
        """Append an entry to the current JSONL file."""
        try:
            log_file = self._get_current_file()
            with open(log_file, "a") as fh:
                fh.write(json.dumps(asdict(entry), default=str) + "\n")
        except OSError as exc:
            logger.error("JSONL write failed: %s", exc)

    def _send_to_kafka(self, entry: AuditEntry) -> None:
        """Produce an entry to the Kafka audit topic (best-effort)."""
        if self._kafka_producer is None:
            return

        try:
            value = json.dumps(asdict(entry), default=str).encode("utf-8")
            self._kafka_producer.produce(
                topic=self._kafka_topic,
                key=entry.event_id.encode("utf-8"),
                value=value,
                callback=self._kafka_delivery_callback,
            )
            # Trigger delivery of buffered messages without blocking
            self._kafka_producer.poll(0)
        except Exception as exc:
            logger.warning("Kafka produce failed (JSONL fallback active): %s", exc)

    @staticmethod
    def _kafka_delivery_callback(err: Any, msg: Any) -> None:
        """Confluent-Kafka delivery report callback."""
        if err is not None:
            logger.warning("Kafka delivery failed: %s", err)
        else:
            logger.debug(
                "Kafka delivery OK: topic=%s partition=%s offset=%s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def _load_chain_state(self) -> None:
        """Restore hash-chain head from existing JSONL files."""
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        if not log_files:
            return

        last_line: Optional[str] = None
        for log_file in reversed(log_files):
            try:
                with open(log_file) as fh:
                    for line in fh:
                        stripped = line.strip()
                        if stripped:
                            last_line = stripped
            except OSError:
                continue
            if last_line:
                break

        if last_line is None:
            return

        try:
            data = json.loads(last_line)
            self._previous_hash = data.get("entry_hash", "genesis")

            # Count total entries across all files
            total = 0
            for lf in log_files:
                try:
                    with open(lf) as fh:
                        total += sum(1 for ln in fh if ln.strip())
                except OSError:
                    continue
            self._entry_count = total
        except json.JSONDecodeError:
            logger.warning("Could not parse last JSONL line; starting fresh chain")

    def get_statistics(self) -> Dict[str, Any]:
        """Return audit log statistics for monitoring dashboards."""
        log_files = list(self.log_dir.glob("audit_*.jsonl"))
        total_size = sum(f.stat().st_size for f in log_files if f.exists())

        return {
            "total_entries": self._entry_count,
            "log_files": len(log_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "chain_head": (
                self._previous_hash[:16] + "..."
                if self._previous_hash != "genesis"
                else "genesis"
            ),
            "kafka_enabled": self._kafka_available,
            "iceberg_enabled": self._iceberg_enabled,
        }


# -----------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------

_logger_instance: Optional[ProductionAuditLogger] = None
_init_lock = threading.Lock()


def init_audit_logger(
    log_dir: str = "logs/audit",
    max_file_size_mb: int = 100,
) -> ProductionAuditLogger:
    """Initialise the global ``ProductionAuditLogger`` singleton.

    Safe to call multiple times; subsequent calls return the existing
    instance.

    Parameters
    ----------
    log_dir : str
        Directory for JSONL fallback files.
    max_file_size_mb : int
        Maximum JSONL file size before rotation.

    Returns
    -------
    ProductionAuditLogger
    """
    global _logger_instance
    with _init_lock:
        if _logger_instance is None:
            _logger_instance = ProductionAuditLogger(
                log_dir=log_dir,
                max_file_size_mb=max_file_size_mb,
            )
    return _logger_instance


def get_audit_logger() -> ProductionAuditLogger:
    """Return the global ``ProductionAuditLogger``, initialising if needed.

    Returns
    -------
    ProductionAuditLogger
    """
    global _logger_instance
    if _logger_instance is None:
        return init_audit_logger()
    return _logger_instance

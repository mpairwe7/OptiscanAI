"""Offline queue for deferred DHIS2 API submissions.

Queues operations when network is unavailable and flushes with
exponential backoff when connectivity returns.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DHIS2Operation:
    """A queued DHIS2 API operation."""

    operation_id: str
    operation_type: str  # create_referral | create_patient | submit_aggregate
    payload: dict
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    last_error: str = ""


@dataclass
class FlushResult:
    """Result of flushing the offline queue."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    remaining: int = 0


class DHIS2OfflineQueue:
    """File-based offline queue for DHIS2 operations."""

    def __init__(self, queue_dir: str = "data/dhis2_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue_file = self.queue_dir / "pending.jsonl"

    async def enqueue(self, operation: DHIS2Operation) -> str:
        """Write operation to the queue file."""
        entry = {
            "operation_id": operation.operation_id,
            "operation_type": operation.operation_type,
            "payload": operation.payload,
            "created_at": operation.created_at,
            "retry_count": operation.retry_count,
        }
        with open(self._queue_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            "Queued DHIS2 operation: %s (%s)", operation.operation_id, operation.operation_type
        )
        return operation.operation_id

    async def flush(self, client: Any) -> FlushResult:
        """Attempt to submit all queued operations with exponential backoff."""
        if not self._queue_file.exists():
            return FlushResult()

        operations = []
        with open(self._queue_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    operations.append(json.loads(line))

        if not operations:
            return FlushResult()

        result = FlushResult(total=len(operations))
        remaining = []

        for op in operations:
            try:
                success = await self._execute_operation(client, op)
                if success:
                    result.succeeded += 1
                else:
                    op["retry_count"] = op.get("retry_count", 0) + 1
                    remaining.append(op)
                    result.failed += 1
            except Exception as e:
                op["retry_count"] = op.get("retry_count", 0) + 1
                op["last_error"] = str(e)
                remaining.append(op)
                result.failed += 1

                # Exponential backoff: stop after 3 consecutive failures
                if result.failed >= 3:
                    remaining.extend(operations[operations.index(op) + 1 :])
                    break

        # Rewrite queue with remaining items
        result.remaining = len(remaining)
        with open(self._queue_file, "w") as f:
            for op in remaining:
                f.write(json.dumps(op) + "\n")

        logger.info(
            "DHIS2 queue flush: %d/%d succeeded, %d remaining",
            result.succeeded,
            result.total,
            result.remaining,
        )
        return result

    async def _execute_operation(self, client: Any, op: dict) -> bool:
        """Execute a single queued operation."""
        op_type = op.get("operation_type")
        payload = op.get("payload", {})

        from .models import AggregateReport, ReferralEvent

        if op_type == "create_referral":
            event_id = await client.create_referral_event(ReferralEvent(**payload))
            return bool(event_id)
        elif op_type == "submit_aggregate":
            result = await client.submit_aggregate_report(AggregateReport(**payload))
            return result.get("status") != "ERROR"
        else:
            logger.warning("Unknown operation type: %s", op_type)
            return False

    async def get_pending_count(self) -> int:
        """Count of operations waiting to be submitted."""
        if not self._queue_file.exists():
            return 0
        with open(self._queue_file) as f:
            return sum(1 for line in f if line.strip())

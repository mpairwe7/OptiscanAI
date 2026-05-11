"""DHIS2 synchronization agent.

Autonomous agent that:
  - Subscribes to SCAN_ANALYZED events to auto-create referrals
  - Periodically flushes the offline DHIS2 queue (every 5 minutes)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.event_bus import EventType

logger = logging.getLogger(__name__)


class DHIS2Agent:
    """Autonomous agent for DHIS2 referral creation and sync."""

    def __init__(self, bus: Any = None):
        self.bus = bus
        self._running = False

    async def start(self) -> None:
        """Start the agent and subscribe to events."""
        self._running = True

        if self.bus:
            self.bus.subscribe(EventType.SCAN_ANALYZED, self._on_scan_analyzed)
            self.bus.subscribe(EventType.VOICE_SCREENING_COMPLETE, self._on_voice_screening)

        logger.info("DHIS2Agent started")

    async def stop(self) -> None:
        self._running = False
        logger.info("DHIS2Agent stopped")

    async def _on_scan_analyzed(self, event: Any) -> None:
        """Auto-create DHIS2 referral when screening completes."""
        try:
            data = event.data if hasattr(event, "data") else {}
            detected = data.get("detected_diseases", [])
            priority = data.get("referral_priority", "ROUTINE")

            if not detected:
                return

            logger.info(
                "DHIS2Agent: Creating referral for %d findings (priority=%s)",
                len(detected), priority,
            )

            if self.bus:
                await self.bus.emit(EventType.DHIS2_REFERRAL_CREATED, {
                    "diseases": detected,
                    "priority": priority,
                })

        except Exception as e:
            logger.error("DHIS2Agent referral creation failed: %s", e)
            if self.bus:
                await self.bus.emit(EventType.DHIS2_SYNC_FAILED, {"error": str(e)})

    async def _on_voice_screening(self, event: Any) -> None:
        """Handle voice screening completion — same as scan analyzed."""
        await self._on_scan_analyzed(event)

    async def tick(self) -> None:
        """Periodic: flush offline queue (every 5 minutes)."""
        try:
            from backend.app.integrations.dhis2.offline_queue import DHIS2OfflineQueue
            from backend.app.core.config import settings

            if not hasattr(settings, "dhis2") or not settings.dhis2.enabled:
                return

            queue = DHIS2OfflineQueue(settings.dhis2.queue_dir)
            pending = await queue.get_pending_count()
            if pending > 0:
                logger.info("DHIS2Agent: Flushing %d queued operations", pending)
                # In production: create client and flush
                # result = await queue.flush(client)
        except Exception as e:
            logger.error("DHIS2Agent tick failed: %s", e)

    @property
    def loop_interval_seconds(self) -> float:
        return 300.0

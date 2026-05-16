"""In-process async event bus for agent communication.

Events flow between agents without tight coupling. Each agent subscribes
to event types it cares about, and emits events for others to react to.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Screening lifecycle
    SCAN_RECEIVED = "scan.received"
    SCAN_ANALYZED = "scan.analyzed"
    SCAN_FLAGGED = "scan.flagged"
    SCAN_CLEARED = "scan.cleared"

    # Clinical reasoning
    REFERRAL_URGENT = "referral.urgent"
    REFERRAL_EMERGENCY = "referral.emergency"

    # Monitoring
    DRIFT_DETECTED = "monitoring.drift_detected"
    SLA_VIOLATED = "monitoring.sla_violated"
    PERFORMANCE_DEGRADED = "monitoring.performance_degraded"
    ANOMALY_DETECTED = "monitoring.anomaly_detected"

    # Model lifecycle
    RETRAIN_TRIGGERED = "model.retrain_triggered"
    RETRAIN_COMPLETED = "model.retrain_completed"
    MODEL_PROMOTED = "model.promoted"
    MODEL_ROLLED_BACK = "model.rolled_back"

    # Governance
    REVIEW_REQUIRED = "governance.review_required"
    REVIEW_COMPLETED = "governance.review_completed"
    AUDIT_ALERT = "governance.audit_alert"
    COMPLIANCE_CHECK = "governance.compliance_check"

    # Active Learning (Phase 1)
    ACTIVE_LEARNING_SAMPLE_QUEUED = "active_learning.sample_queued"
    FINE_TUNE_TRIGGERED = "active_learning.fine_tune_triggered"
    FINE_TUNE_COMPLETED = "active_learning.fine_tune_completed"

    # Enhanced Monitoring (Phase 1)
    DRIFT_ALERT_TRIGGERED = "monitoring.drift_alert_triggered"

    # Resilience (Phase 2)
    CIRCUIT_BREAKER_OPENED = "resilience.circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "resilience.circuit_breaker_closed"

    # Extended Governance (Phase 3)
    FAIRNESS_EVALUATION_COMPLETED = "governance.fairness_evaluation_completed"
    MODEL_CARD_GENERATED = "governance.model_card_generated"

    # Graceful Degradation (Phase 4)
    GRACEFUL_DEGRADATION_ACTIVATED = "resilience.degradation_activated"

    # Voice (Phase 2)
    VOICE_SESSION_STARTED = "voice.session_started"
    VOICE_TRANSCRIPTION_COMPLETE = "voice.transcription_complete"
    VOICE_SCREENING_COMPLETE = "voice.screening_complete"
    VOICE_REFERRAL_SPOKEN = "voice.referral_spoken"

    # DHIS2 (Phase 3)
    DHIS2_REFERRAL_CREATED = "dhis2.referral_created"
    DHIS2_SYNC_COMPLETED = "dhis2.sync_completed"
    DHIS2_SYNC_FAILED = "dhis2.sync_failed"

    # Payments (Phase 3)
    PAYMENT_REQUESTED = "payment.requested"
    PAYMENT_COMPLETED = "payment.completed"
    PAYMENT_FAILED = "payment.failed"

    # SMS/USSD (Phase 3)
    SMS_REFERRAL_SENT = "sms.referral_sent"
    SMS_DELIVERY_CONFIRMED = "sms.delivery_confirmed"
    USSD_SESSION_COMPLETED = "ussd.session_completed"

    # Privacy (Phase 3)
    CONSENT_RECORDED = "privacy.consent_recorded"
    CONSENT_REVOKED = "privacy.consent_revoked"

    # System
    AGENT_STARTED = "system.agent_started"
    AGENT_STOPPED = "system.agent_stopped"
    AGENT_ERROR = "system.agent_error"
    HEARTBEAT = "system.heartbeat"


@dataclass
class Event:
    type: EventType
    source: str  # agent or component that emitted
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    event_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())[:8]


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub event bus for agent coordination.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.SCAN_ANALYZED, my_handler)
        await bus.emit(Event(type=EventType.SCAN_ANALYZED, source="screening_agent", data={...}))
    """

    def __init__(self, max_history: int = 500):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history
        self._running = False
        self._external_transport = None  # Phase 2: Kafka transport

    def set_external_transport(self, transport) -> None:
        """Set an external transport (e.g., Kafka) for durable event delivery.

        The transport must implement an async ``publish(event: Event)`` method.
        In-process dispatch continues regardless; external transport is additive.
        """
        self._external_transport = transport
        logger.info(f"External event transport set: {type(transport).__name__}")

    def subscribe(self, event_type: EventType, handler: EventHandler):
        """Register a handler for an event type."""
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """Remove a handler."""
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h is not handler
        ]

    async def emit(self, event: Event):
        """Dispatch event to all subscribers. Non-blocking: handler errors are logged, not raised."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers = self._handlers.get(event.type, [])
        if not handlers:
            logger.debug(f"Event {event.type.value} from {event.source} — no subscribers")
            return

        logger.info(
            f"Event {event.type.value} from {event.source} -> {len(handlers)} handler(s)",
            extra={"event_id": event.event_id},
        )

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    f"Handler {handler.__qualname__} failed on {event.type.value}: {e}",
                    exc_info=True,
                )

        # Phase 2: Publish to external transport (Kafka) if configured
        if self._external_transport is not None:
            try:
                await self._external_transport.publish(event)
            except Exception as e:
                logger.error(f"External transport publish failed: {e}")

    def get_history(
        self,
        event_type: EventType | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """Query recent event history."""
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        return events[-limit:]

    @property
    def stats(self) -> dict:
        """Event bus statistics for monitoring."""
        type_counts: dict[str, int] = defaultdict(int)
        for e in self._history:
            type_counts[e.type.value] += 1
        return {
            "total_events": len(self._history),
            "subscriber_count": sum(len(h) for h in self._handlers.values()),
            "event_types_seen": dict(type_counts),
        }


# Global singleton
event_bus = EventBus()

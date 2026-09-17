"""Tests for NATS JetStream and Event Transports."""

import pytest

from src.agents.event_bus import Event, EventBus, EventType
from src.agents.event_transports import (
    NATSJetStreamTransport,
    create_transport_from_config,
)


@pytest.mark.asyncio
async def test_nats_transport_offline_standby_mode():
    """Test NATS transport operates gracefully when no daemon is listening."""
    transport = NATSJetStreamTransport(servers=["nats://127.0.0.1:4222"])
    await transport.connect()

    event = Event(
        type=EventType.SCAN_ANALYZED,
        source="screening_agent",
        data={"case_id": "test_123", "priority": "ROUTINE"},
    )
    # Publish should not raise exception even when broker is offline
    await transport.publish(event)
    await transport.close()


@pytest.mark.asyncio
async def test_event_bus_with_external_transport():
    """Test EventBus properly invokes registered external transport."""
    bus = EventBus(max_history=10)

    dispatched = []

    class MockTransport:
        async def publish(self, ev):
            dispatched.append(ev)

    bus.set_external_transport(MockTransport())

    test_event = Event(
        type=EventType.SCAN_RECEIVED,
        source="test_source",
        data={"payload": "sample"},
    )
    await bus.emit(test_event)

    assert len(dispatched) == 1
    assert dispatched[0].event_id == test_event.event_id


def test_create_transport_from_config(monkeypatch):
    monkeypatch.setenv("EVENT_BROKER_TYPE", "nats")
    transport = create_transport_from_config()
    assert isinstance(transport, NATSJetStreamTransport)

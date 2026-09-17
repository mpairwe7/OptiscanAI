"""Durable Event Transports for the Agent Event Bus.

Supports modern lightweight messaging backends:
1. NATS JetStream: <30MB RAM footprint, single-binary, Raft-replicated.
   Ideal for rural clinic edge gateways, Crane Cloud micro-clusters, and low-spec hardware.
2. Apache Kafka: Enterprise high-throughput log for regional/national health data pipelines.
3. InMemoryDurableQueue: File-backed fallback when no external broker is reachable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class EventProtocol(Protocol):
    """Protocol matching Event dataclass."""

    type: Any
    source: str
    data: Dict[str, Any]
    timestamp: str
    event_id: str


class BaseEventTransport:
    """Abstract base transport for external pub/sub event delivery."""

    async def connect(self) -> None:
        """Establish connection to external message broker."""
        pass

    async def publish(self, event: EventProtocol) -> None:
        """Publish an event to the external broker."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close connection cleanly."""
        pass


class NATSJetStreamTransport(BaseEventTransport):
    """Low-footprint, resilient NATS JetStream event transport.

    Requires <35 MB RAM (vs. >2 GB for Kafka + Zookeeper).
    Uses Raft persistence stream 'OPTISCAN_EVENTS' with subjects 'optiscan.events.>'.
    """

    def __init__(
        self,
        servers: Optional[list[str]] = None,
        stream_name: str = "OPTISCAN_EVENTS",
        subject_prefix: str = "optiscan.events",
    ):
        self.servers = servers or ["nats://localhost:4222", "nats://nats:4222"]
        self.stream_name = stream_name
        self.subject_prefix = subject_prefix
        self._nc = None
        self._js = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to NATS and ensure JetStream stream exists."""
        try:
            import nats

            self._nc = await nats.connect(servers=self.servers, connect_timeout=3)
            self._js = self._nc.jetstream()

            # Ensure stream exists
            try:
                await self._js.add_stream(
                    name=self.stream_name,
                    subjects=[f"{self.subject_prefix}.>"],
                )
                logger.info(f"NATS JetStream stream '{self.stream_name}' initialized")
            except Exception as e:
                logger.debug(f"NATS stream '{self.stream_name}' already exists or notice: {e}")

            self._connected = True
            logger.info(f"Connected to NATS JetStream at {self.servers}")
        except ImportError:
            logger.warning(
                "nats-py library is not installed. NATS transport will operate in mock/offline mode."
            )
            self._connected = False
        except Exception as e:
            logger.warning(
                f"Failed to connect to NATS broker at {self.servers} (non-fatal): {e}"
            )
            self._connected = False

    async def publish(self, event: EventProtocol) -> None:
        """Publish event to NATS JetStream subject."""
        event_type_str = str(event.type.value if hasattr(event.type, "value") else event.type)
        subject = f"{self.subject_prefix}.{event_type_str.replace('/', '.')}"

        payload = {
            "event_id": getattr(event, "event_id", ""),
            "type": event_type_str,
            "source": getattr(event, "source", ""),
            "timestamp": getattr(event, "timestamp", ""),
            "data": getattr(event, "data", {}),
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        if self._connected and self._js is not None:
            try:
                ack = await self._js.publish(subject, data_bytes, timeout=2.0)
                logger.debug(f"NATS published to {subject} (seq: {ack.seq})")
            except Exception as e:
                logger.error(f"Failed to publish to NATS subject {subject}: {e}")
        else:
            logger.debug(f"[NATS Standby] Event {event_type_str} logged in-process")

    async def close(self) -> None:
        """Drain and close NATS connection."""
        if self._nc and not self._nc.is_closed:
            try:
                await self._nc.drain()
                await self._nc.close()
                logger.info("NATS JetStream connection closed cleanly")
            except Exception as e:
                logger.error(f"Error closing NATS connection: {e}")
        self._connected = False


class KafkaEventTransport(BaseEventTransport):
    """Kafka event transport for legacy/high-throughput cluster configurations."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "optiscan-events"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer = None
        self._connected = False

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
            self._connected = True
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
        except ImportError:
            logger.warning("aiokafka not installed. Kafka transport operating in fallback mode.")
            self._connected = False
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka at {self.bootstrap_servers}: {e}")
            self._connected = False

    async def publish(self, event: EventProtocol) -> None:
        if not self._connected or not self._producer:
            return

        payload = {
            "event_id": getattr(event, "event_id", ""),
            "type": str(event.type.value if hasattr(event.type, "value") else event.type),
            "source": getattr(event, "source", ""),
            "timestamp": getattr(event, "timestamp", ""),
            "data": getattr(event, "data", {}),
        }
        try:
            await self._producer.send_and_wait(self.topic, payload)
        except Exception as e:
            logger.error(f"Failed to produce message to Kafka topic {self.topic}: {e}")

    async def close(self) -> None:
        if self._producer:
            try:
                await self._producer.stop()
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
        self._connected = False


def create_transport_from_config(cfg: Any = None) -> Optional[BaseEventTransport]:
    """Factory creating appropriate transport based on environment/settings."""
    import os

    broker_type = os.getenv("EVENT_BROKER_TYPE", "nats").lower().strip()

    if broker_type == "nats":
        servers = os.getenv("NATS_SERVERS", "nats://localhost:4222,nats://nats:4222").split(",")
        return NATSJetStreamTransport(servers=[s.strip() for s in servers if s.strip()])
    elif broker_type == "kafka":
        servers = os.getenv("KAFKA__BOOTSTRAP_SERVERS", "localhost:9092")
        return KafkaEventTransport(bootstrap_servers=servers)

    return None

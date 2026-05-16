"""WebhookEvent — idempotency log for provider webhooks."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Enum, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_pk
from backend.app.models.subscription import PaymentProvider


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[str] = uuid_pk()
    provider: Mapped[PaymentProvider] = mapped_column(
        Enum(PaymentProvider, name="payment_provider", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    provider_event_id: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    received_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    __table_args__ = (
        UniqueConstraint("provider", "provider_event_id", name="uq_webhook_provider_event"),
    )

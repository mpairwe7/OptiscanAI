"""PaymentIntent — tracks a pending payment with a provider."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Enum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk
from backend.app.models.subscription import PaymentProvider


class PaymentIntentStatus(str, enum.Enum):
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class PaymentIntent(Base):
    __tablename__ = "payment_intents"

    id: Mapped[str] = uuid_pk()
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False, index=True)
    subscription_id: Mapped[Optional[str]] = uuid_fk("subscriptions.id", nullable=True)
    invoice_id: Mapped[Optional[str]] = uuid_fk("invoices.id", nullable=True)

    provider: Mapped[PaymentProvider] = mapped_column(
        Enum(
            PaymentProvider,
            name="payment_provider",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    provider_intent_id: Mapped[Optional[str]] = mapped_column(
        String(200), unique=True, nullable=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)

    status: Mapped[PaymentIntentStatus] = mapped_column(
        Enum(
            PaymentIntentStatus,
            name="payment_intent_status",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=PaymentIntentStatus.REQUIRES_ACTION,
        nullable=False,
    )

    # For MoMo: caller's phone number; for Stripe: the customer
    phone_msisdn: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    plan_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    billing_cycle: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    raw_callback: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)

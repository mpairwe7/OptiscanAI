"""Subscription model — one per Organization."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Enum, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk

if TYPE_CHECKING:
    from backend.app.models.organization import Organization
    from backend.app.models.plan import Plan


class SubscriptionStatus(str, enum.Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"


class BillingCycle(str, enum.Enum):
    MONTHLY = "monthly"
    ANNUAL = "annual"


class PaymentProvider(str, enum.Enum):
    STRIPE = "stripe"
    MTN = "mtn"
    AIRTEL = "airtel"
    FLUTTERWAVE = "flutterwave"
    MANUAL = "manual"  # Free tier + comped accounts


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = uuid_pk()
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False)
    plan_id: Mapped[str] = uuid_fk("plans.id", nullable=False)

    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(
            SubscriptionStatus,
            name="subscription_status",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=SubscriptionStatus.ACTIVE,
        nullable=False,
    )
    billing_cycle: Mapped[BillingCycle] = mapped_column(
        Enum(BillingCycle, name="billing_cycle", values_callable=lambda x: [e.value for e in x]),
        default=BillingCycle.MONTHLY,
        nullable=False,
    )
    provider: Mapped[PaymentProvider] = mapped_column(
        Enum(
            PaymentProvider, name="payment_provider", values_callable=lambda x: [e.value for e in x]
        ),
        default=PaymentProvider.MANUAL,
        nullable=False,
    )

    # Usage window — the quota check uses this period
    current_period_start: Mapped[datetime] = mapped_column(
        TimestampTZ, default=utcnow, nullable=False
    )
    current_period_end: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)

    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    canceled_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    trial_ends_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)

    # Provider IDs (for reconciliation)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Phase F: extra seats bought beyond the plan's included seat_limit.
    # Effective seat limit = plan.seat_limit + additional_seats.
    additional_seats: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Stripe Subscription Item id for the extra-seat line — set when the org
    # first buys extra seats so subsequent quantity changes hit the same item.
    stripe_seat_item_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=utcnow, onupdate=utcnow, nullable=False
    )

    organization: Mapped["Organization"] = relationship(back_populates="subscription")
    plan: Mapped["Plan"] = relationship()

    __table_args__ = (
        UniqueConstraint("organization_id", name="uq_subscription_organization"),
        Index("ix_subscription_status_period_end", "status", "current_period_end"),
    )

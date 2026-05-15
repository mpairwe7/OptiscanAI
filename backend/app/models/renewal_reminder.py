"""RenewalReminder — log of reminder emails sent for MoMo/Flutterwave subs.

Used to prevent double-sending and to expose a CSV audit for ops.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Enum, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk


class ReminderKind(str, enum.Enum):
    D7 = "7d"
    D3 = "3d"
    D1 = "1d"
    EXPIRED = "expired"


class RenewalReminder(Base):
    __tablename__ = "renewal_reminders"

    id: Mapped[str] = uuid_pk()
    subscription_id: Mapped[str] = uuid_fk("subscriptions.id", nullable=False, index=True)
    kind: Mapped[ReminderKind] = mapped_column(
        Enum(ReminderKind, name="renewal_reminder_kind", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    # The period_end timestamp this reminder was sent against — lets us send
    # a fresh set of reminders for the NEXT period after a renewal happens.
    period_end: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    sent_to: Mapped[str] = mapped_column(String(320), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    __table_args__ = (
        # One reminder per (subscription, period_end, kind). Renewing the
        # subscription advances period_end so a new set of reminders is allowed.
        UniqueConstraint(
            "subscription_id", "period_end", "kind",
            name="uq_renewal_reminder_sub_period_kind",
        ),
    )

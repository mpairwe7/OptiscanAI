"""Invoice model — one per billing period (or one per ad-hoc MoMo payment)."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Enum, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk
from backend.app.models.subscription import PaymentProvider


class InvoiceStatus(str, enum.Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = uuid_pk()
    subscription_id: Mapped[str] = uuid_fk("subscriptions.id", nullable=False)
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False, index=True)

    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)

    status: Mapped[InvoiceStatus] = mapped_column(
        Enum(InvoiceStatus, name="invoice_status", values_callable=lambda x: [e.value for e in x]),
        default=InvoiceStatus.DRAFT,
        nullable=False,
    )
    provider: Mapped[PaymentProvider] = mapped_column(
        Enum(PaymentProvider, name="payment_provider", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    provider_invoice_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    hosted_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    period_start: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    period_end: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)

    issued_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    paid_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)

    __table_args__ = (
        Index("ix_invoice_org_issued", "organization_id", "issued_at"),
    )

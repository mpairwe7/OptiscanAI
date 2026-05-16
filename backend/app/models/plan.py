"""Plan catalog — Free, Clinician, Practice, Health System."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_pk


class PlanCode(str, enum.Enum):
    FREE = "free"
    CLINICIAN = "clinician"
    PRACTICE = "practice"
    HEALTH_SYSTEM = "health_system"


# Numeric rank for tier comparisons in feature_gate
TIER_RANK: dict[str, int] = {
    PlanCode.FREE.value: 0,
    PlanCode.CLINICIAN.value: 1,
    PlanCode.PRACTICE.value: 2,
    PlanCode.HEALTH_SYSTEM.value: 3,
}


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[str] = uuid_pk()
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    tagline: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Pricing — cents, USD. None for contact-sales tier.
    monthly_price_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    annual_price_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)

    # Limits (None = unlimited)
    scan_limit_monthly: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    seat_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    is_contact_sales: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Feature flags surfaced in pricing UI + comparison matrix
    features: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Stripe price IDs (set by ops after creating prices in Stripe dashboard)
    stripe_price_id_monthly: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stripe_price_id_annual: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, onupdate=utcnow, nullable=False)

    @property
    def rank(self) -> int:
        return TIER_RANK.get(self.code, 0)

"""UsageEvent — one row per billable action.

This is the hot path for quota enforcement. The composite index
(organization_id, event_type, occurred_at) makes the monthly COUNT(*)
query O(log n) even at millions of rows.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Enum, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk


class UsageEventType(str, enum.Enum):
    SCAN = "scan"
    EXPLAIN_GRADCAM = "explain_gradcam"
    EXPLAIN_LIME = "explain_lime"
    EXPLAIN_SHAP = "explain_shap"
    EXPLAIN_IG = "explain_ig"
    EXPLAIN_ELI5 = "explain_eli5"
    CLINICAL_REASONING = "clinical_reasoning"
    AUDIT_EXPORT = "audit_export"


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False)
    user_id: Mapped[Optional[str]] = uuid_fk("users.id", nullable=True)

    event_type: Mapped[UsageEventType] = mapped_column(
        Enum(
            UsageEventType, name="usage_event_type", values_callable=lambda x: [e.value for e in x]
        ),
        nullable=False,
    )
    quantity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    request_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)

    __table_args__ = (
        # Hot path: monthly quota query.
        Index("ix_usage_org_type_occurred", "organization_id", "event_type", "occurred_at"),
        Index("ix_usage_occurred", "occurred_at"),
    )

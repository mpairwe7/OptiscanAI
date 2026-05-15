"""Shared SQLAlchemy column types and helpers."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy.dialects.postgresql import UUID as PGUUID


def uuid_pk():
    """Standard UUID v4 primary-key column."""
    return mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


def uuid_fk(target: str, *, nullable: bool = False, **kw):
    from sqlalchemy import ForeignKey
    return mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(target, ondelete="CASCADE"),
        nullable=nullable,
        **kw,
    )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


TimestampTZ = DateTime(timezone=True)

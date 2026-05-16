"""Membership: links a User to an Organization with a role."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk

if TYPE_CHECKING:
    from backend.app.models.organization import Organization
    from backend.app.models.user import User


class MembershipRole(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    CLINICIAN = "clinician"
    VIEWER = "viewer"


class MembershipStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"


class Membership(Base):
    __tablename__ = "memberships"

    id: Mapped[str] = uuid_pk()
    user_id: Mapped[str] = uuid_fk("users.id", nullable=False, index=True)
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False, index=True)

    role: Mapped[MembershipRole] = mapped_column(
        Enum(MembershipRole, name="membership_role", values_callable=lambda x: [e.value for e in x]),
        default=MembershipRole.CLINICIAN,
        nullable=False,
    )
    status: Mapped[MembershipStatus] = mapped_column(
        Enum(MembershipStatus, name="membership_status", values_callable=lambda x: [e.value for e in x]),
        default=MembershipStatus.ACTIVE,
        nullable=False,
        index=True,
    )

    invited_by_user_id: Mapped[Optional[str]] = uuid_fk("users.id", nullable=True)
    invited_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="memberships", foreign_keys=[user_id])
    organization: Mapped["Organization"] = relationship(back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
    )

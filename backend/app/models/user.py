"""User account model."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_pk

if TYPE_CHECKING:
    from backend.app.models.membership import Membership
    from backend.app.models.organization import Organization


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    email_normalized: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    email_verified_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Onboarding metadata (collected post-signup, optional)
    practitioner_role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)  # ISO-3166 alpha-2
    facility_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, onupdate=utcnow, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        # Membership has two FKs to users.id (user_id + invited_by_user_id);
        # pin the relationship to the primary one.
        foreign_keys="Membership.user_id",
    )
    owned_organizations: Mapped[list["Organization"]] = relationship(
        back_populates="owner",
        foreign_keys="Organization.owner_user_id",
    )

    __table_args__ = (
        Index("ix_users_email_normalized_active", "email_normalized", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"

"""Organization (tenant) model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk

if TYPE_CHECKING:
    from backend.app.models.membership import Membership
    from backend.app.models.subscription import Subscription
    from backend.app.models.user import User


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = uuid_pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    billing_email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)

    owner_user_id: Mapped[str] = uuid_fk("users.id", nullable=False)

    # When True, this is the auto-created personal org for a Free/Clinician user.
    # Practice+ orgs are explicitly created and can host multiple seats.
    is_personal: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=utcnow, onupdate=utcnow, nullable=False
    )

    owner: Mapped["User"] = relationship(
        back_populates="owned_organizations", foreign_keys=[owner_user_id]
    )
    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    subscription: Mapped[Optional["Subscription"]] = relationship(
        back_populates="organization",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Organization {self.slug}>"

"""Short-lived tokens: refresh, email-verify, password-reset, magic-link, invite."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Enum, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.db import Base
from backend.app.models._types import TimestampTZ, utcnow, uuid_fk, uuid_pk
from backend.app.models.membership import MembershipRole


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = uuid_pk()
    user_id: Mapped[str] = uuid_fk("users.id", nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    device_fingerprint: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)

    __table_args__ = (Index("ix_refresh_user_revoked", "user_id", "revoked_at"),)


class EmailVerificationToken(Base):
    __tablename__ = "email_verification_tokens"

    id: Mapped[str] = uuid_pk()
    user_id: Mapped[str] = uuid_fk("users.id", nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = uuid_pk()
    user_id: Mapped[str] = uuid_fk("users.id", nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)


class MagicLinkToken(Base):
    __tablename__ = "magic_link_tokens"

    id: Mapped[str] = uuid_pk()
    # email (not user_id) — magic link can sign up a user who doesn't yet exist
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)


class OrganizationInvite(Base):
    __tablename__ = "organization_invites"

    id: Mapped[str] = uuid_pk()
    organization_id: Mapped[str] = uuid_fk("organizations.id", nullable=False, index=True)
    invited_by_user_id: Mapped[str] = uuid_fk("users.id", nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    role: Mapped[MembershipRole] = mapped_column(
        Enum(
            MembershipRole,
            name="membership_role",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=MembershipRole.CLINICIAN,
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TimestampTZ, nullable=False)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TimestampTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampTZ, default=utcnow, nullable=False)

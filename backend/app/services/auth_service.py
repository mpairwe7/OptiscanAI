"""Auth flows: register, login, refresh, magic-link, email-verify, password reset."""
from __future__ import annotations

import logging
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import (
    encode_access_token,
    generate_email_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from backend.app.models.membership import Membership, MembershipRole, MembershipStatus
from backend.app.models.organization import Organization
from backend.app.models.plan import Plan, PlanCode
from backend.app.models.subscription import (
    BillingCycle,
    PaymentProvider,
    Subscription,
    SubscriptionStatus,
)
from backend.app.models.tokens import (
    EmailVerificationToken,
    MagicLinkToken,
    PasswordResetToken,
    RefreshToken,
)
from backend.app.models.user import User
from backend.app.services import email_templates
from backend.app.services.email_service import send_rendered

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_EMAIL_NORMALIZE_RE = re.compile(r"\s+")


def normalize_email(email: str) -> str:
    return _EMAIL_NORMALIZE_RE.sub("", email).strip().lower()


def slugify(name: str) -> str:
    base = _SLUG_RE.sub("-", name.lower()).strip("-") or "org"
    return f"{base}-{secrets.token_hex(3)}"


# ── Tokens ──

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def issue_token_pair(
    db: AsyncSession,
    user: User,
    organization: Organization,
    role: str,
) -> tuple[str, str, int]:
    """Return (access_token, refresh_token, refresh_expires_seconds)."""
    access = encode_access_token(
        user_id=str(user.id),
        org_id=str(organization.id),
        role=role,
        ttl_seconds=settings.jwt_access_ttl_seconds,
    )
    refresh_plain = generate_refresh_token()
    refresh_row = RefreshToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(refresh_plain),
        expires_at=_utcnow() + timedelta(seconds=settings.jwt_refresh_ttl_seconds),
    )
    db.add(refresh_row)
    await db.flush()
    return access, refresh_plain, settings.jwt_refresh_ttl_seconds


async def revoke_refresh_token(db: AsyncSession, token: str) -> None:
    stmt = select(RefreshToken).where(RefreshToken.token_hash == hash_token(token))
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row and row.revoked_at is None:
        row.revoked_at = _utcnow()


async def rotate_refresh_token(db: AsyncSession, presented: str) -> tuple[User, Organization, str, str]:
    stmt = select(RefreshToken).where(RefreshToken.token_hash == hash_token(presented))
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None or row.revoked_at is not None or row.expires_at < _utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    user = await db.get(User, row.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
    membership = (
        await db.execute(
            select(Membership)
            .where(Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE)
            .limit(1)
        )
    ).scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No active organization")
    organization = await db.get(Organization, membership.organization_id)

    # Revoke old, issue new
    row.revoked_at = _utcnow()
    access, refresh, _ttl = await issue_token_pair(db, user, organization, membership.role.value)
    return user, organization, access, refresh


# ── Registration & personal org bootstrap ──

async def get_plan_by_code(db: AsyncSession, code: str) -> Plan:
    plan = (await db.execute(select(Plan).where(Plan.code == code))).scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=500, detail=f"Plan {code!r} not seeded — run alembic upgrade head")
    return plan


async def register_user(
    db: AsyncSession,
    *,
    email: str,
    password: Optional[str],
    full_name: Optional[str],
) -> tuple[User, Organization]:
    """Create user + personal org + Free subscription."""
    email_norm = normalize_email(email)
    existing = (
        await db.execute(select(User).where(User.email_normalized == email_norm))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        id=uuid.uuid4(),
        email=email.strip(),
        email_normalized=email_norm,
        password_hash=hash_password(password) if password else None,
        full_name=full_name,
    )
    db.add(user)
    await db.flush()

    free_plan = await get_plan_by_code(db, PlanCode.FREE.value)
    org = Organization(
        id=uuid.uuid4(),
        name=full_name or email_norm.split("@")[0],
        slug=slugify(full_name or email_norm.split("@")[0]),
        billing_email=email_norm,
        owner_user_id=user.id,
        is_personal=True,
    )
    db.add(org)
    await db.flush()

    membership = Membership(
        id=uuid.uuid4(),
        user_id=user.id,
        organization_id=org.id,
        role=MembershipRole.OWNER,
        status=MembershipStatus.ACTIVE,
        accepted_at=_utcnow(),
    )
    db.add(membership)

    period_end = _utcnow() + timedelta(days=settings.billing.free_period_days)
    sub = Subscription(
        id=uuid.uuid4(),
        organization_id=org.id,
        plan_id=free_plan.id,
        status=SubscriptionStatus.ACTIVE,
        billing_cycle=BillingCycle.MONTHLY,
        provider=PaymentProvider.MANUAL,
        current_period_start=_utcnow(),
        current_period_end=period_end,
    )
    db.add(sub)
    await db.flush()

    return user, org


# ── Email verification ──

async def issue_email_verification(db: AsyncSession, user: User) -> str:
    token = generate_email_token()
    db.add(EmailVerificationToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=_utcnow() + timedelta(seconds=settings.email.verification_link_ttl_seconds),
    ))
    await send_rendered(
        to=user.email,
        email=email_templates.email_verification(full_name=user.full_name, token=token),
    )
    return token


async def consume_email_verification(db: AsyncSession, token: str) -> User:
    row = (
        await db.execute(
            select(EmailVerificationToken).where(
                EmailVerificationToken.token_hash == hash_token(token),
            )
        )
    ).scalar_one_or_none()
    if row is None or row.used_at is not None or row.expires_at < _utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    row.used_at = _utcnow()
    user = await db.get(User, row.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.email_verified_at = _utcnow()
    return user


# ── Password reset ──

async def issue_password_reset(db: AsyncSession, email: str) -> Optional[str]:
    email_norm = normalize_email(email)
    user = (
        await db.execute(select(User).where(User.email_normalized == email_norm))
    ).scalar_one_or_none()
    if user is None:
        return None  # silent — don't leak account existence
    token = generate_email_token()
    db.add(PasswordResetToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=_utcnow() + timedelta(seconds=settings.email.password_reset_ttl_seconds),
    ))
    await send_rendered(
        to=user.email,
        email=email_templates.password_reset(full_name=user.full_name, token=token),
    )
    return token


async def consume_password_reset(db: AsyncSession, token: str, new_password: str) -> User:
    row = (
        await db.execute(
            select(PasswordResetToken).where(PasswordResetToken.token_hash == hash_token(token))
        )
    ).scalar_one_or_none()
    if row is None or row.used_at is not None or row.expires_at < _utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    user = await db.get(User, row.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    row.used_at = _utcnow()
    user.password_hash = hash_password(new_password)
    return user


# ── Magic link ──

async def issue_magic_link(db: AsyncSession, email: str) -> str:
    email_norm = normalize_email(email)
    token = generate_email_token()
    db.add(MagicLinkToken(
        id=uuid.uuid4(),
        email=email_norm,
        token_hash=hash_token(token),
        expires_at=_utcnow() + timedelta(seconds=settings.email.magic_link_ttl_seconds),
    ))
    await send_rendered(
        to=email,
        email=email_templates.magic_link(email=email_norm, token=token),
    )
    return token


async def consume_magic_link(db: AsyncSession, token: str) -> tuple[User, Organization, bool]:
    """Return (user, org, was_created). Creates user + personal org if first sign-in."""
    row = (
        await db.execute(select(MagicLinkToken).where(MagicLinkToken.token_hash == hash_token(token)))
    ).scalar_one_or_none()
    if row is None or row.used_at is not None or row.expires_at < _utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired magic link")
    row.used_at = _utcnow()

    user = (
        await db.execute(select(User).where(User.email_normalized == row.email))
    ).scalar_one_or_none()
    if user is not None:
        membership = (
            await db.execute(
                select(Membership)
                .where(Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE)
                .limit(1)
            )
        ).scalar_one_or_none()
        if membership is None:
            raise HTTPException(status_code=403, detail="User has no active organization")
        org = await db.get(Organization, membership.organization_id)
        user.email_verified_at = user.email_verified_at or _utcnow()
        user.last_login_at = _utcnow()
        return user, org, False

    user, org = await register_user(db, email=row.email, password=None, full_name=None)
    user.email_verified_at = _utcnow()
    user.last_login_at = _utcnow()
    return user, org, True


# ── Password login ──

async def authenticate_password(db: AsyncSession, email: str, password: str) -> tuple[User, Organization, str]:
    email_norm = normalize_email(email)
    user = (
        await db.execute(select(User).where(User.email_normalized == email_norm))
    ).scalar_one_or_none()
    if user is None or user.password_hash is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    membership = (
        await db.execute(
            select(Membership)
            .where(Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE)
            .limit(1)
        )
    ).scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=403, detail="No active organization")
    org = await db.get(Organization, membership.organization_id)
    user.last_login_at = _utcnow()
    return user, org, membership.role.value

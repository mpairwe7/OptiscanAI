"""Organization + membership operations."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import generate_email_token, hash_password, hash_token
from backend.app.models.membership import Membership, MembershipRole, MembershipStatus
from backend.app.models.organization import Organization
from backend.app.models.plan import Plan
from backend.app.models.subscription import Subscription
from backend.app.models.tokens import OrganizationInvite
from backend.app.models.user import User
from backend.app.services import email_templates
from backend.app.services.auth_service import normalize_email, slugify
from backend.app.services.email_service import send_rendered

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def list_user_orgs(db: AsyncSession, user_id: str) -> list[tuple[Organization, str]]:
    stmt = (
        select(Membership, Organization)
        .join(Organization, Membership.organization_id == Organization.id)
        .where(Membership.user_id == user_id, Membership.status == MembershipStatus.ACTIVE)
    )
    rows = (await db.execute(stmt)).all()
    return [(o, m.role.value) for (m, o) in rows]


async def create_organization(db: AsyncSession, owner: User, name: str) -> Organization:
    org = Organization(
        id=uuid.uuid4(),
        name=name,
        slug=slugify(name),
        billing_email=owner.email_normalized,
        owner_user_id=owner.id,
        is_personal=False,
    )
    db.add(org)
    await db.flush()

    # Owner membership
    db.add(Membership(
        id=uuid.uuid4(),
        user_id=owner.id,
        organization_id=org.id,
        role=MembershipRole.OWNER,
        status=MembershipStatus.ACTIVE,
        accepted_at=_utcnow(),
    ))

    # Default to Free subscription — owner upgrades via /billing
    from backend.app.models.plan import PlanCode
    free_plan = (
        await db.execute(select(Plan).where(Plan.code == PlanCode.FREE.value))
    ).scalar_one()
    db.add(Subscription(
        id=uuid.uuid4(),
        organization_id=org.id,
        plan_id=free_plan.id,
        current_period_start=_utcnow(),
        current_period_end=_utcnow() + timedelta(days=settings.billing.free_period_days),
    ))
    await db.flush()
    return org


async def get_active_membership(db: AsyncSession, user_id: str, org_id: str) -> Optional[Membership]:
    return (
        await db.execute(
            select(Membership).where(
                Membership.user_id == user_id,
                Membership.organization_id == org_id,
                Membership.status == MembershipStatus.ACTIVE,
            )
        )
    ).scalar_one_or_none()


async def require_org_role(
    db: AsyncSession, user_id: str, org_id: str, required: set[MembershipRole],
) -> Membership:
    m = await get_active_membership(db, user_id, org_id)
    if m is None or m.role not in required:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return m


async def count_active_seats(db: AsyncSession, org_id: str) -> int:
    stmt = select(func.count()).select_from(Membership).where(
        Membership.organization_id == org_id,
        Membership.status == MembershipStatus.ACTIVE,
    )
    return int((await db.execute(stmt)).scalar_one())


async def invite_member(
    db: AsyncSession,
    *,
    organization: Organization,
    inviter: User,
    email: str,
    role: MembershipRole,
) -> tuple[OrganizationInvite, str]:
    # Check effective seat limit = plan.seat_limit + subscription.additional_seats
    sub = (
        await db.execute(select(Subscription).where(Subscription.organization_id == organization.id))
    ).scalar_one_or_none()
    plan = await db.get(Plan, sub.plan_id) if sub else None
    included = plan.seat_limit if plan else 1
    additional = sub.additional_seats if sub else 0
    effective_limit = None if included is None else included + additional
    if effective_limit is not None:
        used = await count_active_seats(db, str(organization.id))
        pending = (
            await db.execute(
                select(func.count()).select_from(OrganizationInvite).where(
                    OrganizationInvite.organization_id == organization.id,
                    OrganizationInvite.accepted_at.is_(None),
                    OrganizationInvite.revoked_at.is_(None),
                )
            )
        ).scalar_one()
        if used + pending >= effective_limit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "seat_limit_reached",
                    "message": (
                        f"Seat limit reached on the {plan.display_name} plan "
                        f"({effective_limit} total; {additional} extras already purchased)."
                    ),
                    "plan": {
                        "code": plan.code,
                        "included_seats": included,
                        "additional_seats": additional,
                        "effective_seat_limit": effective_limit,
                    },
                    "usage": {"seats_used": used, "seats_pending": pending},
                    "upgrade_url": "/app/team",
                },
            )

    token = generate_email_token()
    invite = OrganizationInvite(
        id=uuid.uuid4(),
        organization_id=organization.id,
        invited_by_user_id=inviter.id,
        email=normalize_email(email),
        role=role,
        token_hash=hash_token(token),
        expires_at=_utcnow() + timedelta(seconds=settings.email.invite_ttl_seconds),
    )
    db.add(invite)
    await db.flush()

    await send_rendered(
        to=email,
        email=email_templates.org_invite(
            inviter_name=inviter.full_name,
            inviter_email=inviter.email,
            organization_name=organization.name,
            role=role.value,
            token=token,
        ),
    )
    return invite, token


async def accept_invite(
    db: AsyncSession,
    *,
    token: str,
    full_name: Optional[str],
    password: Optional[str],
) -> tuple[User, Organization]:
    invite = (
        await db.execute(
            select(OrganizationInvite).where(OrganizationInvite.token_hash == hash_token(token))
        )
    ).scalar_one_or_none()
    if (
        invite is None
        or invite.accepted_at is not None
        or invite.revoked_at is not None
        or invite.expires_at < _utcnow()
    ):
        raise HTTPException(status_code=400, detail="Invalid or expired invite")

    user = (
        await db.execute(select(User).where(User.email_normalized == invite.email))
    ).scalar_one_or_none()
    if user is None:
        if not password:
            raise HTTPException(status_code=400, detail="New users must supply a password")
        user = User(
            id=uuid.uuid4(),
            email=invite.email,
            email_normalized=invite.email,
            password_hash=hash_password(password),
            full_name=full_name,
            email_verified_at=_utcnow(),
        )
        db.add(user)
        await db.flush()

    existing = await get_active_membership(db, str(user.id), str(invite.organization_id))
    if existing is None:
        db.add(Membership(
            id=uuid.uuid4(),
            user_id=user.id,
            organization_id=invite.organization_id,
            role=invite.role,
            status=MembershipStatus.ACTIVE,
            accepted_at=_utcnow(),
        ))
    invite.accepted_at = _utcnow()

    org = await db.get(Organization, invite.organization_id)
    return user, org

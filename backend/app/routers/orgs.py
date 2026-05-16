"""Organization management routes: list/create orgs, invite members, manage roles."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, get_auth_context
from backend.app.core.db import get_db
from backend.app.models.membership import Membership, MembershipRole, MembershipStatus
from backend.app.models.organization import Organization
from backend.app.models.tokens import OrganizationInvite
from backend.app.models.user import User
from backend.app.schemas.orgs import (
    AcceptInviteRequest,
    CreateOrgRequest,
    InviteMemberRequest,
    InviteResponse,
    MemberResponse,
    OrgResponse,
    UpdateMemberRoleRequest,
)
from backend.app.services.org_service import (
    accept_invite,
    create_organization,
    invite_member,
    list_user_orgs,
    require_org_role,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/orgs", tags=["orgs"])

ADMIN_ROLES = {MembershipRole.OWNER, MembershipRole.ADMIN}


@router.get("", response_model=list[OrgResponse])
async def list_orgs(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[OrgResponse]:
    rows = await list_user_orgs(db, str(ctx.user.id))
    return [
        OrgResponse(
            id=str(o.id),
            name=o.name,
            slug=o.slug,
            is_personal=o.is_personal,
            is_active=o.is_active,
            role=role,
            created_at=o.created_at,
        )
        for o, role in rows
    ]


@router.post("", response_model=OrgResponse, status_code=status.HTTP_201_CREATED)
async def create_org(
    body: CreateOrgRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> OrgResponse:
    org = await create_organization(db, ctx.user, body.name)
    return OrgResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        is_personal=org.is_personal,
        is_active=org.is_active,
        role="owner",
        created_at=org.created_at,
    )


@router.get("/{org_id}/members", response_model=list[MemberResponse])
async def list_members(
    org_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[MemberResponse]:
    await require_org_role(
        db,
        str(ctx.user.id),
        org_id,
        ADMIN_ROLES | {MembershipRole.CLINICIAN, MembershipRole.VIEWER},
    )
    stmt = (
        select(Membership, User)
        .join(User, Membership.user_id == User.id)
        .where(
            Membership.organization_id == org_id,
            Membership.status != MembershipStatus.REVOKED,
        )
    )
    rows = (await db.execute(stmt)).all()
    return [
        MemberResponse(
            user_id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            role=m.role.value,
            status=m.status.value,
            joined_at=m.accepted_at,
        )
        for (m, u) in rows
    ]


@router.post(
    "/{org_id}/invites", response_model=InviteResponse, status_code=status.HTTP_201_CREATED
)
async def create_invite(
    org_id: str,
    body: InviteMemberRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> InviteResponse:
    await require_org_role(db, str(ctx.user.id), org_id, ADMIN_ROLES)
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    invite, _token = await invite_member(
        db,
        organization=org,
        inviter=ctx.user,
        email=body.email,
        role=MembershipRole(body.role),
    )
    return InviteResponse(
        id=str(invite.id),
        email=invite.email,
        role=invite.role.value,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
    )


@router.get("/{org_id}/invites", response_model=list[InviteResponse])
async def list_pending_invites(
    org_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[InviteResponse]:
    await require_org_role(db, str(ctx.user.id), org_id, ADMIN_ROLES)
    rows = (
        (
            await db.execute(
                select(OrganizationInvite).where(
                    OrganizationInvite.organization_id == org_id,
                    OrganizationInvite.accepted_at.is_(None),
                    OrganizationInvite.revoked_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    return [
        InviteResponse(
            id=str(i.id),
            email=i.email,
            role=i.role.value,
            expires_at=i.expires_at,
            created_at=i.created_at,
        )
        for i in rows
    ]


@router.post("/{org_id}/invites/{invite_id}/revoke")
async def revoke_invite(
    org_id: str,
    invite_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await require_org_role(db, str(ctx.user.id), org_id, ADMIN_ROLES)
    from datetime import datetime, timezone

    invite = await db.get(OrganizationInvite, invite_id)
    if invite is None or str(invite.organization_id) != org_id:
        raise HTTPException(status_code=404, detail="Invite not found")
    invite.revoked_at = datetime.now(timezone.utc)
    return {"status": "revoked"}


@router.post("/invites/accept")
async def accept_invite_route(
    body: AcceptInviteRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    user, org = await accept_invite(
        db,
        token=body.token,
        full_name=body.full_name,
        password=body.password,
    )
    return {
        "status": "ok",
        "organization": {"id": str(org.id), "name": org.name, "slug": org.slug},
        "user": {"id": str(user.id), "email": user.email},
    }


@router.patch("/{org_id}/members/{user_id}")
async def update_member_role(
    org_id: str,
    user_id: str,
    body: UpdateMemberRoleRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await require_org_role(db, str(ctx.user.id), org_id, ADMIN_ROLES)
    target = (
        await db.execute(
            select(Membership).where(
                Membership.organization_id == org_id,
                Membership.user_id == user_id,
                Membership.status == MembershipStatus.ACTIVE,
            )
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Member not found")
    if target.role == MembershipRole.OWNER:
        raise HTTPException(status_code=400, detail="Cannot change owner role")
    target.role = MembershipRole(body.role)
    return {"status": "ok", "role": target.role.value}


@router.delete("/{org_id}/members/{user_id}")
async def remove_member(
    org_id: str,
    user_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await require_org_role(db, str(ctx.user.id), org_id, ADMIN_ROLES)
    target = (
        await db.execute(
            select(Membership).where(
                Membership.organization_id == org_id,
                Membership.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Member not found")
    if target.role == MembershipRole.OWNER:
        raise HTTPException(status_code=400, detail="Cannot remove owner")
    target.status = MembershipStatus.REVOKED
    target.accepted_at = target.accepted_at  # keep history
    return {"status": "revoked"}

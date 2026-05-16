"""Authentication routes: register / login / refresh / magic-link / verify / password reset / me.

The legacy ``POST /token`` endpoint is kept for backwards compatibility with
existing internal callers; new callers should use ``/login`` instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, create_access_token, get_auth_context
from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.models.organization import Organization
from backend.app.models.plan import Plan
from backend.app.models.subscription import Subscription
from backend.app.models.user import User
from backend.app.schemas.auth import (
    AuthSuccessResponse,
    LoginRequest,
    MagicLinkRequest,
    MeResponse,
    OrgSummary,
    PasswordForgotRequest,
    PasswordResetRequest,
    PlanSummary,
    RegisterRequest,
    SubscriptionSummary,
)
from backend.app.services.auth_service import (
    authenticate_password,
    consume_email_verification,
    consume_magic_link,
    consume_password_reset,
    issue_email_verification,
    issue_magic_link,
    issue_password_reset,
    issue_token_pair,
    register_user,
    revoke_refresh_token,
    rotate_refresh_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Helpers ──


def _set_auth_cookies(response: Response, *, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        key="os_access",
        value=access_token,
        max_age=settings.jwt_access_ttl_seconds,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key="os_refresh",
        value=refresh_token,
        max_age=settings.jwt_refresh_ttl_seconds,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(key="os_access", path="/")
    response.delete_cookie(key="os_refresh", path="/")


# Back-compat shim — older callers still reference these names.
_set_refresh_cookie = lambda r, t, _ttl: r.set_cookie(  # noqa: E731
    key="os_refresh",
    value=t,
    max_age=settings.jwt_refresh_ttl_seconds,
    httponly=True,
    secure=settings.environment == "production",
    samesite="lax",
    path="/",
)
_clear_refresh_cookie = _clear_auth_cookies


async def _build_me_response(
    db: AsyncSession,
    user: User,
    org: Organization,
    role: str,
) -> MeResponse:
    sub: Optional[Subscription] = (
        await db.execute(select(Subscription).where(Subscription.organization_id == org.id))
    ).scalar_one_or_none()
    plan: Optional[Plan] = await db.get(Plan, sub.plan_id) if sub else None
    sub_summary = None
    if sub and plan:
        sub_summary = SubscriptionSummary(
            plan=PlanSummary(
                code=plan.code,
                display_name=plan.display_name,
                scan_limit_monthly=plan.scan_limit_monthly,
                seat_limit=plan.seat_limit,
            ),
            status=sub.status.value,
            billing_cycle=sub.billing_cycle.value,
            current_period_start=sub.current_period_start,
            current_period_end=sub.current_period_end,
            cancel_at_period_end=sub.cancel_at_period_end,
        )
    return MeResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        email_verified=user.email_verified_at is not None,
        is_superuser=bool(user.is_superuser),
        organization=OrgSummary(
            id=str(org.id),
            name=org.name,
            slug=org.slug,
            is_personal=org.is_personal,
        ),
        role=role,
        subscription=sub_summary,
    )


async def _issue_session(
    db: AsyncSession,
    response: Response,
    user: User,
    org: Organization,
    role: str,
) -> AuthSuccessResponse:
    access, refresh, _refresh_ttl = await issue_token_pair(db, user, org, role)
    _set_auth_cookies(response, access_token=access, refresh_token=refresh)
    me = await _build_me_response(db, user, org, role)
    return AuthSuccessResponse(
        user=me,
        access_token=access,
        expires_in=settings.jwt_access_ttl_seconds,
    )


# ── Register ──


@router.post("/register", response_model=AuthSuccessResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthSuccessResponse:
    user, org = await register_user(
        db,
        email=body.email,
        password=body.password,
        full_name=body.full_name,
    )
    try:
        await issue_email_verification(db, user)
    except Exception as exc:
        logger.warning("Failed to send verification email: %s", exc)
    return await _issue_session(db, response, user, org, "owner")


# ── Login ──


@router.post("/login", response_model=AuthSuccessResponse)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthSuccessResponse:
    user, org, role = await authenticate_password(db, body.email, body.password)
    return await _issue_session(db, response, user, org, role)


# ── Refresh ──


@router.post("/refresh", response_model=AuthSuccessResponse)
async def refresh(
    response: Response,
    db: AsyncSession = Depends(get_db),
    os_refresh: str | None = Cookie(default=None),
) -> AuthSuccessResponse:
    if not os_refresh:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    user, org, access, new_refresh = await rotate_refresh_token(db, os_refresh)
    _set_auth_cookies(response, access_token=access, refresh_token=new_refresh)

    from backend.app.models.membership import Membership, MembershipStatus

    membership = (
        await db.execute(
            select(Membership).where(
                Membership.user_id == user.id,
                Membership.organization_id == org.id,
                Membership.status == MembershipStatus.ACTIVE,
            )
        )
    ).scalar_one_or_none()
    role = membership.role.value if membership else "viewer"

    me = await _build_me_response(db, user, org, role)
    return AuthSuccessResponse(
        user=me,
        access_token=access,
        expires_in=settings.jwt_access_ttl_seconds,
    )


# ── Logout ──


@router.post("/logout")
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    os_refresh: str | None = Cookie(default=None),
) -> dict:
    if os_refresh:
        await revoke_refresh_token(db, os_refresh)
    _clear_auth_cookies(response)
    return {"status": "ok"}


# ── Magic link ──


@router.post("/magic-link/request")
async def magic_link_request(body: MagicLinkRequest, db: AsyncSession = Depends(get_db)) -> dict:
    await issue_magic_link(db, body.email)
    return {"status": "ok"}


@router.get("/magic-link/verify", response_model=AuthSuccessResponse)
async def magic_link_verify(
    token: str,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthSuccessResponse:
    user, org, _created = await consume_magic_link(db, token)
    return await _issue_session(db, response, user, org, "owner")


# ── Email verification ──


@router.get("/verify-email")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)) -> dict:
    user = await consume_email_verification(db, token)
    return {"status": "verified", "email": user.email}


@router.post("/verify-email/resend")
async def resend_verification(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if ctx.user.email_verified_at is not None:
        return {"status": "already_verified"}
    await issue_email_verification(db, ctx.user)
    return {"status": "sent"}


# ── Password reset ──


@router.post("/password/forgot")
async def password_forgot(body: PasswordForgotRequest, db: AsyncSession = Depends(get_db)) -> dict:
    await issue_password_reset(db, body.email)
    return {"status": "ok"}


@router.post("/password/reset")
async def password_reset(body: PasswordResetRequest, db: AsyncSession = Depends(get_db)) -> dict:
    await consume_password_reset(db, body.token, body.new_password)
    return {"status": "ok"}


# ── Current user ──


@router.get("/me", response_model=MeResponse)
async def me(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> MeResponse:
    return await _build_me_response(db, ctx.user, ctx.organization, ctx.role)


# ── Legacy token endpoint (backwards compat for on-prem deployments) ──


class _LegacyTokenRequest(BaseModel):
    api_key: str


class _LegacyTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=_LegacyTokenResponse, include_in_schema=False)
async def legacy_token(request: _LegacyTokenRequest):
    """Legacy: exchange a shared API key for a JWT. Preserved for existing scripts.

    New integrations should use POST /login or POST /register.
    """
    if not settings.auth_enabled:
        return _LegacyTokenResponse(
            access_token=create_access_token("anonymous", role="admin"),
            expires_in=settings.jwt_access_ttl_seconds,
        )
    if request.api_key != settings.jwt_secret:
        raise HTTPException(status_code=401, detail="Invalid API key")
    token = create_access_token(
        subject="api_user",
        role="user",
        expires_seconds=settings.jwt_access_ttl_seconds,
    )
    return _LegacyTokenResponse(
        access_token=token,
        expires_in=settings.jwt_access_ttl_seconds,
    )

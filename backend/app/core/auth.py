"""JWT authentication for API endpoints.

Backwards compatible: when `settings.auth_enabled=False`, `get_current_user`
returns an anonymous TokenPayload as before so existing non-SaaS deployments
keep working.

When auth+DB are enabled, `get_auth_context` (new) loads the User row and
the active Organization+Subscription, exposing them as `AuthContext` for
billing-aware endpoints.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.core.security import decode_access_token, encode_access_token

if TYPE_CHECKING:
    from backend.app.models.membership import Membership
    from backend.app.models.organization import Organization
    from backend.app.models.subscription import Subscription
    from backend.app.models.user import User

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class TokenPayload(BaseModel):
    """Backwards-compatible legacy payload (used by routers that don't care about billing)."""
    sub: str
    exp: float
    role: str = "user"
    org: Optional[str] = None


@dataclass
class AuthContext:
    """Rich context for billing-aware endpoints."""
    user: "User"
    membership: "Membership"
    organization: "Organization"
    subscription: Optional["Subscription"]
    role: str


# ── Token issuing ──

def create_access_token(
    subject: str,
    role: str = "user",
    expires_seconds: Optional[int] = None,
    org_id: Optional[str] = None,
) -> str:
    """Backwards-compatible: callers in old auth router still call this."""
    return encode_access_token(
        user_id=subject,
        org_id=org_id,
        role=role,
        ttl_seconds=expires_seconds,
    )


# ── Anonymous fallback (single-tenant on-prem) ──

def _anonymous_payload() -> TokenPayload:
    return TokenPayload(sub="anonymous", exp=time.time() + 3600, role="admin")


# ── Resolve token from header OR httpOnly cookie ──

def _extract_token(
    credentials: Optional[HTTPAuthorizationCredentials],
    cookie_token: Optional[str],
) -> Optional[str]:
    if credentials is not None and credentials.credentials:
        return credentials.credentials
    return cookie_token


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    os_access: Optional[str] = Cookie(default=None),
) -> Optional[TokenPayload]:
    """Legacy dependency — returns lightweight TokenPayload.

    Preserved so existing routers (predict, explain, governance, etc.) keep
    compiling. New billing-aware endpoints should depend on `get_auth_context`.
    """
    if not settings.auth_enabled:
        return _anonymous_payload()

    token = _extract_token(credentials, os_access)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization")

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return TokenPayload(
        sub=str(payload.get("sub", "anonymous")),
        exp=float(payload.get("exp", 0)),
        role=str(payload.get("role", "user")),
        org=payload.get("org"),
    )


async def get_auth_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    os_access: Optional[str] = Cookie(default=None),
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    """Resolve token → User + active Membership + Organization + Subscription.

    Used by billing-aware endpoints. Always requires auth (no anonymous mode).
    """
    from backend.app.models.membership import Membership, MembershipStatus
    from backend.app.models.user import User

    token = _extract_token(credentials, os_access)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization")
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id = payload.get("sub")
    org_id = payload.get("org")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing sub")

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    # Resolve active membership — explicit org from token, else first active membership
    stmt = (
        select(Membership)
        .where(Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE)
        .options(selectinload(Membership.organization))
    )
    if org_id:
        stmt = stmt.where(Membership.organization_id == org_id)
    stmt = stmt.limit(1)
    membership: Membership | None = (await db.execute(stmt)).scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No active organization")

    organization = membership.organization

    # Resolve subscription separately (avoid eager-load explosion)
    from backend.app.models.subscription import Subscription
    sub_stmt = (
        select(Subscription)
        .where(Subscription.organization_id == organization.id)
        .options(selectinload(Subscription.plan))
    )
    subscription: Subscription | None = (await db.execute(sub_stmt)).scalar_one_or_none()

    return AuthContext(
        user=user,
        membership=membership,
        organization=organization,
        subscription=subscription,
        role=membership.role.value,
    )


def require_role(required_role: str):
    """Backwards-compatible role check on legacy TokenPayload."""
    async def role_checker(user: TokenPayload = Depends(get_current_user)):
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth required")
        if user.role != required_role and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user
    return role_checker


async def require_superuser(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """Dependency that admits only platform superusers (ops). Returns the context."""
    if not ctx.user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser required")
    return ctx

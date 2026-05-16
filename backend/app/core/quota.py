"""Quota enforcement dependency for metered endpoints.

Returns a ``BillingContext`` on success; raises HTTPException(402) with a
structured ``quota_exceeded`` payload + ``X-Usage-*`` headers on failure so
the frontend can open the paywall modal.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, get_auth_context
from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.models.plan import PlanCode
from backend.app.models.subscription import Subscription
from backend.app.models.usage_event import UsageEventType
from backend.app.services.usage_service import count_events_in_period

logger = logging.getLogger(__name__)


@dataclass
class BillingContext:
    auth: AuthContext
    used: int
    limit: Optional[int]  # None = unlimited
    period_end_iso: str

    @property
    def remaining(self) -> Optional[int]:
        if self.limit is None:
            return None
        return max(0, self.limit - self.used)


# Tiny in-process cache — avoids re-querying COUNT(*) on bursts.
# Key: (org_id, period_end_iso) -> (count, expires_ts)
_quota_cache: dict[tuple[str, str], tuple[int, float]] = {}


def _cache_get(org_id: str, period_end_iso: str) -> Optional[int]:
    key = (org_id, period_end_iso)
    entry = _quota_cache.get(key)
    if entry is None:
        return None
    count, expires = entry
    if time.time() > expires:
        _quota_cache.pop(key, None)
        return None
    return count


def _cache_put(org_id: str, period_end_iso: str, count: int) -> None:
    _quota_cache[(org_id, period_end_iso)] = (
        count,
        time.time() + settings.billing.quota_cache_ttl_s,
    )


def _cache_bust(org_id: str) -> None:
    for k in list(_quota_cache.keys()):
        if k[0] == org_id:
            _quota_cache.pop(k, None)


def _build_402_payload(
    *,
    plan_code: str,
    scan_limit: int,
    used: int,
    period_end: str,
    recommended: str,
) -> dict:
    return {
        "error": "quota_exceeded",
        "message": f"Monthly scan limit reached on the {plan_code.replace('_', ' ').title()} plan.",
        "plan": {"code": plan_code, "scan_limit_monthly": scan_limit},
        "usage": {"used": used, "limit": scan_limit, "resets_at": period_end},
        "upgrade_url": "/pricing",
        "recommended_plan": recommended,
    }


def _recommend_next_tier(current: str) -> str:
    ladder = [PlanCode.FREE.value, PlanCode.CLINICIAN.value, PlanCode.PRACTICE.value, PlanCode.HEALTH_SYSTEM.value]
    try:
        idx = ladder.index(current)
        return ladder[min(idx + 1, len(ladder) - 1)]
    except ValueError:
        return PlanCode.CLINICIAN.value


async def enforce_scan_quota(
    response: Response,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> BillingContext:
    sub: Optional[Subscription] = ctx.subscription
    if sub is None:
        raise HTTPException(status_code=403, detail="No active subscription")
    plan = sub.plan
    limit = plan.scan_limit_monthly  # None = unlimited

    period_end_iso = sub.current_period_end.isoformat()
    org_id = str(ctx.organization.id)

    used = _cache_get(org_id, period_end_iso)
    if used is None:
        used = await count_events_in_period(
            db,
            organization_id=org_id,
            event_type=UsageEventType.SCAN,
            period_start=sub.current_period_start,
            period_end=sub.current_period_end,
        )
        _cache_put(org_id, period_end_iso, used)

    if limit is not None:
        response.headers["X-Usage-Used"] = str(used)
        response.headers["X-Usage-Limit"] = str(limit)
        response.headers["X-Usage-Resets"] = period_end_iso
        if used >= limit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=_build_402_payload(
                    plan_code=plan.code,
                    scan_limit=limit,
                    used=used,
                    period_end=period_end_iso,
                    recommended=_recommend_next_tier(plan.code),
                ),
                headers={
                    "X-Usage-Used": str(used),
                    "X-Usage-Limit": str(limit),
                    "X-Usage-Resets": period_end_iso,
                },
            )

    return BillingContext(auth=ctx, used=used, limit=limit, period_end_iso=period_end_iso)


def invalidate_quota_cache(org_id: str) -> None:
    """Call after a plan change so the next request reflects the new limit."""
    _cache_bust(org_id)


def increment_cached_count(org_id: str, period_end_iso: str, by: int = 1) -> None:
    """Optimistically bump the cached count after a successful billable event."""
    key = (org_id, period_end_iso)
    entry = _quota_cache.get(key)
    if entry is not None:
        _quota_cache[key] = (entry[0] + by, entry[1])


# ── Inline helpers (used inside endpoints whose primary Depends is legacy) ──

async def check_scan_quota_inline(request, response) -> Optional["BillingContext"]:
    """Resolve auth + quota manually for endpoints that keep their legacy
    Depends(get_current_user) signature (e.g. /api/v1/predict).

    Returns None when billing or DB is disabled. Raises HTTPException(402)
    when the org is over its monthly scan quota; the response body matches
    the structure consumed by the frontend PaywallModal.
    """
    from sqlalchemy import select

    from backend.app.core.db import session_factory
    from backend.app.core.security import decode_access_token
    from backend.app.models.membership import Membership, MembershipStatus
    from backend.app.models.organization import Organization
    from backend.app.models.subscription import Subscription
    from backend.app.models.user import User

    if not settings.billing.enabled:
        return None
    factory = session_factory()
    if factory is None:
        return None

    # Pull token from Authorization header OR os_access cookie
    auth_header = request.headers.get("authorization", "")
    token: Optional[str] = None
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("os_access")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    user_id = payload.get("sub")
    org_id = payload.get("org")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")

    async with factory() as db:
        user = await db.get(User, user_id)
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found")

        stmt = select(Membership).where(
            Membership.user_id == user.id, Membership.status == MembershipStatus.ACTIVE,
        )
        if org_id:
            stmt = stmt.where(Membership.organization_id == org_id)
        membership = (await db.execute(stmt.limit(1))).scalar_one_or_none()
        if membership is None:
            raise HTTPException(status_code=403, detail="No active organization")
        organization = await db.get(Organization, membership.organization_id)

        sub = (
            await db.execute(select(Subscription).where(Subscription.organization_id == organization.id))
        ).scalar_one_or_none()
        if sub is None:
            raise HTTPException(status_code=403, detail="No active subscription")

        from backend.app.services.billing_service import roll_period_forward_if_needed
        await roll_period_forward_if_needed(db, sub)

        from backend.app.models.plan import Plan
        plan = await db.get(Plan, sub.plan_id)
        period_end_iso = sub.current_period_end.isoformat()

        from backend.app.models.usage_event import UsageEventType
        from backend.app.services.usage_service import count_events_in_period
        used = _cache_get(str(organization.id), period_end_iso)
        if used is None:
            used = await count_events_in_period(
                db,
                organization_id=str(organization.id),
                event_type=UsageEventType.SCAN,
                period_start=sub.current_period_start,
                period_end=sub.current_period_end,
            )
            _cache_put(str(organization.id), period_end_iso, used)

        limit = plan.scan_limit_monthly
        if limit is not None:
            response.headers["X-Usage-Used"] = str(used)
            response.headers["X-Usage-Limit"] = str(limit)
            response.headers["X-Usage-Resets"] = period_end_iso
            if used >= limit:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=_build_402_payload(
                        plan_code=plan.code,
                        scan_limit=limit,
                        used=used,
                        period_end=period_end_iso,
                        recommended=_recommend_next_tier(plan.code),
                    ),
                    headers={
                        "X-Usage-Used": str(used),
                        "X-Usage-Limit": str(limit),
                        "X-Usage-Resets": period_end_iso,
                    },
                )

        await db.commit()
        return _SimpleCtx(
            organization_id=str(organization.id),
            user_id=str(user.id),
            period_end_iso=period_end_iso,
            used=used,
            limit=limit,
        )


@dataclass
class _SimpleCtx:
    organization_id: str
    user_id: str
    period_end_iso: str
    used: int
    limit: Optional[int]


async def record_scan_usage(ctx: "_SimpleCtx", request_id: Optional[str] = None) -> None:
    """Insert a UsageEvent row for a successful scan."""
    from backend.app.core.db import session_factory
    from backend.app.models.usage_event import UsageEvent, UsageEventType

    factory = session_factory()
    if factory is None:
        return
    async with factory() as db:
        db.add(UsageEvent(
            organization_id=ctx.organization_id,
            user_id=ctx.user_id,
            event_type=UsageEventType.SCAN,
            quantity=1,
            request_id=request_id,
        ))
        await db.commit()
    increment_cached_count(ctx.organization_id, ctx.period_end_iso, by=1)

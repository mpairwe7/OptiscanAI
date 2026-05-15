"""Tier-based feature gate dependency.

Use as ``Depends(require_tier("clinician"))`` on a router endpoint. Returns
the underlying :class:`AuthContext` on success; raises HTTP 403 with a
``feature_locked`` payload on failure so the frontend can show the upsell
sheet with the cheapest unlocking tier pre-selected.
"""
from __future__ import annotations

from typing import Callable, Optional

from fastapi import Depends, HTTPException, status

from backend.app.core.auth import AuthContext, get_auth_context
from backend.app.core.config import settings
from backend.app.models.plan import TIER_RANK, PlanCode


def _build_403_payload(*, required: str, current: str, feature: str) -> dict:
    return {
        "error": "feature_locked",
        "message": (
            f"This feature requires the {required.replace('_', ' ').title()} plan. "
            f"You're currently on {current.replace('_', ' ').title()}."
        ),
        "feature": feature,
        "required_plan": required,
        "current_plan": current,
        "upgrade_url": "/pricing",
    }


def require_tier(min_tier: str, *, feature: str = "") -> Callable:
    if min_tier not in TIER_RANK:
        raise ValueError(f"Unknown tier {min_tier!r}")
    required_rank = TIER_RANK[min_tier]

    async def _dep_billing_off() -> Optional[AuthContext]:
        # On-prem / research mode: no gate.
        return None

    async def _dep_billing_on(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if ctx.subscription is None:
            current_code = PlanCode.FREE.value
        else:
            current_code = ctx.subscription.plan.code
        current_rank = TIER_RANK.get(current_code, 0)
        if current_rank < required_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_build_403_payload(
                    required=min_tier,
                    current=current_code,
                    feature=feature or min_tier,
                ),
            )
        return ctx

    # Capture the setting at factory time — flips on by re-importing app.
    if settings.billing.enabled:
        return _dep_billing_on
    return _dep_billing_off

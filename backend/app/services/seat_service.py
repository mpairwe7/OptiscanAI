"""Practice-tier seat management.

The Practice plan ships with 5 seats. Customers can buy extra seats via Stripe;
each extra seat is a separate Stripe Subscription Item billed at the Practice
extra-seat price.

The effective seat limit is:
    plan.seat_limit + subscription.additional_seats

For non-Stripe rails (MoMo / Flutterwave) we currently surface a 400 — the
user must contact sales. Stripe is the only path with native quantity
manipulation that supports proration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.quota import invalidate_quota_cache
from backend.app.models.organization import Organization
from backend.app.models.plan import Plan, PlanCode
from backend.app.models.subscription import (
    BillingCycle,
    PaymentProvider,
    Subscription,
)

logger = logging.getLogger(__name__)


@dataclass
class SeatState:
    included_seats: int  # plan.seat_limit
    additional_seats: int
    effective_limit: Optional[int]  # None = unlimited
    seats_used: int  # active memberships
    can_buy_more: bool
    per_seat_cents: int  # display cost (for the cycle currently in effect)
    cycle: str


async def get_seat_state(db: AsyncSession, *, subscription: Subscription) -> SeatState:
    plan = await db.get(Plan, subscription.plan_id)
    if plan is None:
        raise HTTPException(status_code=500, detail="Plan missing")

    # Count active seats
    from sqlalchemy import func

    from backend.app.models.membership import Membership, MembershipStatus

    seats_used = int(
        (
            await db.execute(
                select(func.count())
                .select_from(Membership)
                .where(
                    Membership.organization_id == subscription.organization_id,
                    Membership.status == MembershipStatus.ACTIVE,
                )
            )
        ).scalar_one()
    )

    included = plan.seat_limit or 0
    additional = subscription.additional_seats or 0
    effective = None if plan.seat_limit is None else included + additional

    can_buy_more = (
        plan.code == PlanCode.PRACTICE.value
        and subscription.provider == PaymentProvider.STRIPE
        and settings.stripe.enabled
    )

    per_seat_cents = (
        settings.stripe.practice_extra_seat_annual_cents
        if subscription.billing_cycle == BillingCycle.ANNUAL
        else settings.stripe.practice_extra_seat_monthly_cents
    )

    return SeatState(
        included_seats=included,
        additional_seats=additional,
        effective_limit=effective,
        seats_used=seats_used,
        can_buy_more=can_buy_more,
        per_seat_cents=per_seat_cents,
        cycle=subscription.billing_cycle.value,
    )


async def update_seat_quantity(
    db: AsyncSession,
    *,
    organization: Organization,
    target_total_additional_seats: int,
) -> Subscription:
    """Move the org to ``target_total_additional_seats`` extra seats.

    Pre-conditions:
      * Subscription is on Stripe
      * Plan is Practice
      * ``target_total_additional_seats >= 0``
      * Resulting effective limit doesn't drop below current ``seats_used``
    """
    if target_total_additional_seats < 0:
        raise HTTPException(status_code=400, detail="Seat count cannot be negative")

    sub = (
        await db.execute(
            select(Subscription).where(Subscription.organization_id == organization.id)
        )
    ).scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription")

    plan = await db.get(Plan, sub.plan_id)
    if plan is None:
        raise HTTPException(status_code=500, detail="Plan missing")

    if plan.code != PlanCode.PRACTICE.value:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "seats_not_supported",
                "message": "Extra seats are only available on the Practice plan.",
                "upgrade_url": "/pricing",
            },
        )
    if sub.provider != PaymentProvider.STRIPE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "seats_stripe_only",
                "message": (
                    "Self-serve seat purchases require a Stripe card subscription. "
                    "MoMo/Flutterwave subscribers can email sales@makstartup.com to add seats."
                ),
            },
        )
    if not sub.stripe_subscription_id:
        raise HTTPException(status_code=500, detail="Stripe subscription missing")

    state = await get_seat_state(db, subscription=sub)
    new_effective = state.included_seats + target_total_additional_seats
    if new_effective < state.seats_used:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "would_remove_active_seats",
                "message": (
                    f"You have {state.seats_used} active members but the requested "
                    f"plan only supports {new_effective}. Remove members first."
                ),
            },
        )

    from backend.app.integrations import stripe_client

    stripe_sub, seat_item_id = await stripe_client.set_seat_quantity(
        stripe_subscription_id=sub.stripe_subscription_id,
        seat_item_id=sub.stripe_seat_item_id,
        new_quantity=target_total_additional_seats,
        billing_cycle=sub.billing_cycle.value,
    )

    sub.additional_seats = target_total_additional_seats
    sub.stripe_seat_item_id = seat_item_id or None
    invalidate_quota_cache(str(organization.id))

    logger.info(
        "Seat update: org=%s target=%s stripe_sub=%s",
        organization.id,
        target_total_additional_seats,
        stripe_sub.get("id") if isinstance(stripe_sub, dict) else stripe_sub.id,
    )
    return sub

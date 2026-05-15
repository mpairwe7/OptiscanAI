"""Subscription lifecycle: plan changes, cancel/resume, period roll-forward."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.quota import invalidate_quota_cache
from backend.app.models.invoice import Invoice, InvoiceStatus
from backend.app.models.organization import Organization
from backend.app.models.plan import Plan, PlanCode
from backend.app.models.subscription import (
    BillingCycle,
    PaymentProvider,
    Subscription,
    SubscriptionStatus,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def get_active_subscription(db: AsyncSession, org_id: str) -> Optional[Subscription]:
    return (
        await db.execute(select(Subscription).where(Subscription.organization_id == org_id))
    ).scalar_one_or_none()


async def get_plan(db: AsyncSession, code: str) -> Plan:
    plan = (await db.execute(select(Plan).where(Plan.code == code))).scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Unknown plan {code!r}")
    return plan


def _next_period_end(now: datetime, cycle: BillingCycle) -> datetime:
    if cycle == BillingCycle.ANNUAL:
        return now + timedelta(days=365)
    return now + timedelta(days=30)


async def change_plan_immediately(
    db: AsyncSession,
    *,
    organization: Organization,
    plan_code: str,
    cycle: BillingCycle,
    provider: PaymentProvider = PaymentProvider.MANUAL,
) -> Subscription:
    """Switch the org to a new plan, immediately starting a fresh period.

    Free-tier downgrades:
      * Block when the org has more active members than the Free tier supports
        (Free = 1 seat). Forces the customer to remove members first.
      * Cancel the existing Stripe subscription so the customer isn't billed
        next cycle. ``proration_behavior="create_prorations"`` issues a credit
        for the unused time.
      * MoMo/Flutterwave subscriptions: just flip locally; the already-paid
        period is forfeit (the customer initiated the downgrade).
    """
    from backend.app.models.membership import Membership, MembershipStatus
    from sqlalchemy import func

    sub = await get_active_subscription(db, str(organization.id))
    new_plan = await get_plan(db, plan_code)
    if new_plan.is_contact_sales and plan_code != PlanCode.FREE.value:
        raise HTTPException(
            status_code=400,
            detail="Contact-sales plans require a sales conversation — contact sales instead.",
        )

    # Free-tier downgrade guard: don't orphan team members
    if plan_code == PlanCode.FREE.value and new_plan.seat_limit is not None:
        active_members = int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(Membership)
                    .where(
                        Membership.organization_id == organization.id,
                        Membership.status == MembershipStatus.ACTIVE,
                    )
                )
            ).scalar_one()
        )
        if active_members > new_plan.seat_limit:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "members_exceed_target_seats",
                    "message": (
                        f"You have {active_members} active members but the {new_plan.display_name} plan only "
                        f"supports {new_plan.seat_limit}. Remove the extras from /app/team first, then retry."
                    ),
                    "active_members": active_members,
                    "target_plan": plan_code,
                    "target_seat_limit": new_plan.seat_limit,
                },
            )

    # Cancel Stripe billing if the current sub was paid — avoid next-cycle charges
    if (
        plan_code == PlanCode.FREE.value
        and sub is not None
        and sub.provider == PaymentProvider.STRIPE
        and sub.stripe_subscription_id
    ):
        try:
            import stripe
            from backend.app.core.config import settings as _settings
            stripe.api_key = _settings.stripe.api_key
            stripe.Subscription.delete(
                sub.stripe_subscription_id,
                prorate=True,
            )
        except Exception as exc:
            logger.warning(
                "Stripe subscription cancellation failed during downgrade for org=%s: %s",
                organization.id,
                exc,
            )
            # Continue — we still flip the local plan so the customer's UI reflects Free.

    now = _utcnow()
    if sub is None:
        sub = Subscription(
            id=uuid.uuid4(),
            organization_id=organization.id,
            plan_id=new_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=cycle,
            provider=provider,
            current_period_start=now,
            current_period_end=_next_period_end(now, cycle),
        )
        db.add(sub)
    else:
        sub.plan_id = new_plan.id
        sub.billing_cycle = cycle
        sub.provider = provider if plan_code == PlanCode.FREE.value else sub.provider
        sub.status = SubscriptionStatus.ACTIVE
        sub.current_period_start = now
        sub.current_period_end = _next_period_end(now, cycle)
        sub.cancel_at_period_end = False
        sub.canceled_at = None
        if plan_code == PlanCode.FREE.value:
            sub.additional_seats = 0
            sub.stripe_seat_item_id = None
            sub.stripe_subscription_id = None
            # Keep stripe_customer_id so re-upgrades reuse the same Stripe Customer
    await db.flush()
    invalidate_quota_cache(str(organization.id))
    return sub


async def cancel_at_period_end(db: AsyncSession, organization: Organization) -> Subscription:
    sub = await get_active_subscription(db, str(organization.id))
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription to cancel")
    sub.cancel_at_period_end = True
    sub.canceled_at = _utcnow()
    return sub


async def resume_subscription(db: AsyncSession, organization: Organization) -> Subscription:
    sub = await get_active_subscription(db, str(organization.id))
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription to resume")
    sub.cancel_at_period_end = False
    sub.canceled_at = None
    return sub


async def roll_period_forward_if_needed(db: AsyncSession, sub: Subscription) -> None:
    """Free tier (no provider) advances by 30 days on demand."""
    now = _utcnow()
    if sub.current_period_end > now:
        return
    if sub.provider != PaymentProvider.MANUAL:
        return  # paid subs are advanced by webhook
    if sub.cancel_at_period_end:
        # Cancellation took effect — fall back to Free
        sub.status = SubscriptionStatus.CANCELED
        return
    sub.current_period_start = now
    sub.current_period_end = _next_period_end(now, sub.billing_cycle)
    invalidate_quota_cache(str(sub.organization_id))


async def list_invoices(db: AsyncSession, org_id: str, limit: int = 25) -> list[Invoice]:
    stmt = (
        select(Invoice)
        .where(Invoice.organization_id == org_id)
        .order_by(Invoice.issued_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def write_invoice(
    db: AsyncSession,
    *,
    sub: Subscription,
    amount_cents: int,
    provider: PaymentProvider,
    provider_invoice_id: Optional[str] = None,
    hosted_url: Optional[str] = None,
    status: InvoiceStatus = InvoiceStatus.PAID,
) -> Invoice:
    inv = Invoice(
        id=uuid.uuid4(),
        subscription_id=sub.id,
        organization_id=sub.organization_id,
        amount_cents=amount_cents,
        currency="USD",
        status=status,
        provider=provider,
        provider_invoice_id=provider_invoice_id,
        hosted_url=hosted_url,
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
        issued_at=_utcnow(),
        paid_at=_utcnow() if status == InvoiceStatus.PAID else None,
    )
    db.add(inv)
    return inv


# ── Stripe event reconciliation ──

def _from_unix(ts: int | None) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _seat_quantity_from_items(stripe_subscription: dict, extra_seat_price_ids: set[str]) -> tuple[int, Optional[str]]:
    """Inspect Stripe subscription items, return (extra_seat_qty, seat_item_id)."""
    items = stripe_subscription.get("items", {}).get("data", []) or []
    for it in items:
        price_id = (it.get("price") or {}).get("id")
        if price_id and price_id in extra_seat_price_ids:
            return int(it.get("quantity") or 0), it.get("id")
    return 0, None


async def apply_stripe_subscription_event(
    db: AsyncSession,
    *,
    stripe_subscription: dict,
    plan_code: Optional[str] = None,
    billing_cycle: Optional[str] = None,
) -> Optional[Subscription]:
    """Create or update our local Subscription from a Stripe subscription object.

    Accepts a dict (works with stripe-python's dict-like StripeObject) so this
    function is unit-testable without the SDK.
    """
    metadata = stripe_subscription.get("metadata") or {}
    organization_id = metadata.get("organization_id")
    if not organization_id:
        logger.warning("Stripe subscription missing organization_id metadata: %s", stripe_subscription.get("id"))
        return None

    plan_code = plan_code or metadata.get("plan_code")
    if not plan_code:
        logger.warning("Stripe subscription missing plan_code metadata: %s", stripe_subscription.get("id"))
        return None

    plan = await get_plan(db, plan_code)
    sub = await get_active_subscription(db, organization_id)
    if sub is None:
        sub = Subscription(
            id=uuid.uuid4(),
            organization_id=organization_id,
            plan_id=plan.id,
            provider=PaymentProvider.STRIPE,
            current_period_start=_utcnow(),
            current_period_end=_utcnow() + timedelta(days=30),
        )
        db.add(sub)
        await db.flush()

    sub.plan_id = plan.id
    sub.provider = PaymentProvider.STRIPE
    sub.stripe_subscription_id = stripe_subscription.get("id")
    sub.stripe_customer_id = stripe_subscription.get("customer")

    status_map = {
        "active": SubscriptionStatus.ACTIVE,
        "trialing": SubscriptionStatus.TRIALING,
        "past_due": SubscriptionStatus.PAST_DUE,
        "canceled": SubscriptionStatus.CANCELED,
        "unpaid": SubscriptionStatus.PAST_DUE,
        "incomplete": SubscriptionStatus.INCOMPLETE,
        "incomplete_expired": SubscriptionStatus.CANCELED,
    }
    sub.status = status_map.get(stripe_subscription.get("status", ""), sub.status)
    sub.cancel_at_period_end = bool(stripe_subscription.get("cancel_at_period_end"))

    period_start = _from_unix(stripe_subscription.get("current_period_start"))
    period_end = _from_unix(stripe_subscription.get("current_period_end"))
    if period_start:
        sub.current_period_start = period_start
    if period_end:
        sub.current_period_end = period_end

    if billing_cycle:
        sub.billing_cycle = BillingCycle(billing_cycle)
    elif metadata.get("billing_cycle"):
        try:
            sub.billing_cycle = BillingCycle(metadata["billing_cycle"])
        except ValueError:
            pass

    # Sync extra-seat quantity from Stripe items
    from backend.app.core.config import settings as _settings
    seat_price_ids = {
        _settings.stripe.practice_extra_seat_monthly_price_id,
        _settings.stripe.practice_extra_seat_annual_price_id,
    } - {""}
    if seat_price_ids:
        qty, item_id = _seat_quantity_from_items(stripe_subscription, seat_price_ids)
        sub.additional_seats = qty
        if item_id:
            sub.stripe_seat_item_id = item_id

    # The Subscription is the source of truth for plan. Earlier code wrote a
    # denormalized `org.plan_id` here but Organization has no such column;
    # leaving the touch in place would 500 every Stripe webhook.
    invalidate_quota_cache(str(organization_id))
    return sub


async def apply_stripe_invoice_paid(
    db: AsyncSession,
    *,
    stripe_invoice: dict,
) -> Optional[Invoice]:
    """Record a paid Stripe invoice in our Invoice table."""
    organization_id = (stripe_invoice.get("metadata") or {}).get("organization_id")
    subscription_id_stripe = stripe_invoice.get("subscription")

    sub: Optional[Subscription] = None
    if subscription_id_stripe:
        from sqlalchemy import select as _select
        sub = (
            await db.execute(
                _select(Subscription).where(Subscription.stripe_subscription_id == subscription_id_stripe)
            )
        ).scalar_one_or_none()
    if sub is None and organization_id:
        sub = await get_active_subscription(db, organization_id)
    if sub is None:
        logger.warning("Stripe invoice cannot be linked to a subscription: %s", stripe_invoice.get("id"))
        return None

    amount = stripe_invoice.get("amount_paid") or stripe_invoice.get("amount_due") or 0
    inv = Invoice(
        id=uuid.uuid4(),
        subscription_id=sub.id,
        organization_id=sub.organization_id,
        amount_cents=int(amount),
        currency=(stripe_invoice.get("currency") or "usd").upper(),
        status=InvoiceStatus.PAID,
        provider=PaymentProvider.STRIPE,
        provider_invoice_id=stripe_invoice.get("id"),
        hosted_url=stripe_invoice.get("hosted_invoice_url"),
        pdf_url=stripe_invoice.get("invoice_pdf"),
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
        issued_at=_utcnow(),
        paid_at=_utcnow(),
    )
    db.add(inv)
    return inv

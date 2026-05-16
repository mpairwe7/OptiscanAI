"""Renewal reminders for MoMo/Flutterwave subscriptions.

MoMo and Flutterwave don't natively support recurring auto-debit, so we run a
daily cron that finds subscriptions on those rails whose ``current_period_end``
is approaching, and emails the owner with a one-click re-checkout link.

Windows (configurable below):
  - 7 days before period_end
  - 3 days before period_end
  - 1 day before period_end
  - on or after period_end (one final "expired" reminder)

The :class:`RenewalReminder` table records each successfully sent reminder
with a unique constraint on (subscription_id, period_end, kind), so re-running
the cron is a no-op until the subscription's period_end advances after a paid
renewal.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.organization import Organization
from backend.app.models.plan import Plan
from backend.app.models.renewal_reminder import ReminderKind, RenewalReminder
from backend.app.models.subscription import (
    BillingCycle,
    PaymentProvider,
    Subscription,
    SubscriptionStatus,
)
from backend.app.models.user import User
from backend.app.services import email_templates
from backend.app.services.email_service import send_rendered

logger = logging.getLogger(__name__)


_MOMO_PROVIDERS = (
    PaymentProvider.MTN,
    PaymentProvider.AIRTEL,
    PaymentProvider.FLUTTERWAVE,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RenewalRunResult:
    found_subscriptions: int
    reminders_sent: int
    reminders_skipped: int  # already sent for this (sub, period_end, kind)
    errors: int

    def as_dict(self) -> dict:
        return {
            "found_subscriptions": self.found_subscriptions,
            "reminders_sent": self.reminders_sent,
            "reminders_skipped": self.reminders_skipped,
            "errors": self.errors,
        }


def _days_remaining(period_end: datetime, now: datetime) -> int:
    return (period_end - now).days


def _kind_for(period_end: datetime, now: datetime) -> Optional[ReminderKind]:
    """Map a subscription's period_end to the appropriate ReminderKind.

    Returns None if the sub is outside any reminder window.
    """
    delta = period_end - now
    seconds = delta.total_seconds()
    one_day = 86400
    if seconds <= 0 and seconds > -one_day:
        return ReminderKind.EXPIRED
    if 0 < seconds <= one_day:
        return ReminderKind.D1
    if one_day < seconds <= 3 * one_day:
        return ReminderKind.D3
    if 3 * one_day < seconds <= 7 * one_day:
        return ReminderKind.D7
    return None


async def _candidate_subscriptions(db: AsyncSession, now: datetime) -> list[Subscription]:
    """All MoMo/Flutterwave subscriptions inside any reminder window."""
    window_start = now - timedelta(days=1)
    window_end = now + timedelta(days=7)
    stmt = (
        select(Subscription)
        .where(
            Subscription.provider.in_(_MOMO_PROVIDERS),
            Subscription.status.in_(
                [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
            ),
            Subscription.cancel_at_period_end.is_(False),
            Subscription.current_period_end >= window_start,
            Subscription.current_period_end <= window_end,
        )
        .order_by(Subscription.current_period_end)
    )
    return list((await db.execute(stmt)).scalars().all())


async def _already_sent(
    db: AsyncSession,
    *,
    subscription_id: str,
    period_end: datetime,
    kind: ReminderKind,
) -> bool:
    stmt = select(RenewalReminder).where(
        RenewalReminder.subscription_id == subscription_id,
        RenewalReminder.period_end == period_end,
        RenewalReminder.kind == kind,
        RenewalReminder.error.is_(None),
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def send_reminder_for(
    db: AsyncSession,
    *,
    subscription: Subscription,
    kind: ReminderKind,
) -> RenewalReminder:
    """Send one reminder, idempotent on (sub_id, period_end, kind)."""
    plan = await db.get(Plan, subscription.plan_id)
    org = await db.get(Organization, subscription.organization_id)
    owner = await db.get(User, org.owner_user_id) if org else None
    if plan is None or org is None or owner is None:
        raise RuntimeError(
            f"Reminder skipped — missing plan/org/owner for sub {subscription.id}"
        )

    price_cents = (
        plan.annual_price_cents
        if subscription.billing_cycle == BillingCycle.ANNUAL
        else plan.monthly_price_cents
    )
    amount_usd = (price_cents or 0) / 100.0
    days_remaining = max(_days_remaining(subscription.current_period_end, _utcnow()), 0)

    rendered = email_templates.renewal_reminder(
        full_name=owner.full_name,
        organization_name=org.name,
        plan_display_name=plan.display_name,
        period_end_iso=subscription.current_period_end.isoformat(),
        days_remaining=days_remaining,
        plan_code=plan.code,
        billing_cycle=subscription.billing_cycle.value,
        amount_usd=amount_usd,
    )

    error_text: Optional[str] = None
    try:
        await send_rendered(to=owner.email, email=rendered)
    except Exception as exc:  # noqa: BLE001 — log the failure so the row is honest
        logger.exception("Renewal reminder send failed for sub=%s", subscription.id)
        error_text = str(exc)[:2000]

    reminder = RenewalReminder(
        id=uuid.uuid4(),
        subscription_id=subscription.id,
        kind=kind,
        period_end=subscription.current_period_end,
        sent_to=owner.email,
        error=error_text,
    )
    db.add(reminder)
    return reminder


async def run_renewal_reminders(db: AsyncSession) -> RenewalRunResult:
    """Find every subscription in a reminder window and send if not yet sent."""
    now = _utcnow()
    subs = await _candidate_subscriptions(db, now)

    sent = 0
    skipped = 0
    errors = 0

    for sub in subs:
        kind = _kind_for(sub.current_period_end, now)
        if kind is None:
            continue

        if await _already_sent(db, subscription_id=str(sub.id), period_end=sub.current_period_end, kind=kind):
            skipped += 1
            continue

        try:
            reminder = await send_reminder_for(db, subscription=sub, kind=kind)
            if reminder.error is None:
                sent += 1
            else:
                errors += 1
        except Exception:
            logger.exception("Failed to record reminder for sub=%s", sub.id)
            errors += 1

    await db.commit()
    return RenewalRunResult(
        found_subscriptions=len(subs),
        reminders_sent=sent,
        reminders_skipped=skipped,
        errors=errors,
    )

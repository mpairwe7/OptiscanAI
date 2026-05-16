"""Mobile money subscription billing.

MoMo (MTN/Airtel) doesn't natively support recurring auto-debit, so the model
is **one-shot payment for one period**:

  1. User picks plan + cycle on /app/checkout/<plan>?cycle=…
  2. Backend creates a ``PaymentIntent`` (requires_action) with idempotency_key
  3. Backend tells the existing :class:`MobileMoneyClient` to push a USSD prompt
     to the user's phone
  4. User enters PIN on phone, provider POSTs to /momo/callback/<provider>
  5. Callback handler verifies HMAC, finds the intent, advances the
     Subscription period, writes an Invoice row

The Subscription's ``current_period_end`` advances by 30 or 365 days from now
(not from the previous period_end) — this keeps it consistent with our quota
windows. Renewals are user-initiated (we email a reminder + a re-checkout link).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.quota import invalidate_quota_cache
from backend.app.integrations.mobile_money.client import MobileMoneyClient
from backend.app.models.invoice import Invoice, InvoiceStatus
from backend.app.models.organization import Organization
from backend.app.models.payment_intent import PaymentIntent, PaymentIntentStatus
from backend.app.models.plan import Plan
from backend.app.models.subscription import (
    BillingCycle,
    PaymentProvider,
    Subscription,
    SubscriptionStatus,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _provider_enum(provider: str) -> PaymentProvider:
    return {
        "mtn": PaymentProvider.MTN,
        "airtel": PaymentProvider.AIRTEL,
        "flutterwave": PaymentProvider.FLUTTERWAVE,
    }.get(provider, PaymentProvider.MANUAL)


def _period_delta(cycle: BillingCycle) -> timedelta:
    return timedelta(days=365 if cycle == BillingCycle.ANNUAL else 30)


def usd_to_ugx(amount_usd_cents: int) -> int:
    """Convert plan-table USD cents to a UGX whole-shilling amount."""
    rate = settings.mobile_money.ugx_per_usd
    return int(round((amount_usd_cents / 100.0) * rate))


def _client() -> MobileMoneyClient:
    mm = settings.mobile_money
    return MobileMoneyClient(
        mtn_api_key=mm.mtn_api_key,
        mtn_api_secret=mm.mtn_api_secret,
        mtn_subscription_key=mm.mtn_subscription_key,
        mtn_environment=mm.mtn_environment,
        airtel_client_id=mm.airtel_client_id,
        airtel_client_secret=mm.airtel_client_secret,
    )


# ── Initiate ──


async def initiate_momo_payment(
    db: AsyncSession,
    *,
    organization: Organization,
    plan_code: str,
    billing_cycle: BillingCycle,
    phone: str,
    provider: str,  # "mtn" | "airtel"
    user_id: str,
) -> PaymentIntent:
    """Create a PaymentIntent and push a USSD prompt to the user's phone."""
    if not settings.mobile_money.enabled:
        raise HTTPException(status_code=503, detail="Mobile money not enabled")
    if provider not in {"mtn", "airtel"}:
        raise HTTPException(status_code=400, detail="Unsupported MoMo provider")

    plan = (await db.execute(select(Plan).where(Plan.code == plan_code))).scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Unknown plan {plan_code}")
    if plan.is_contact_sales:
        raise HTTPException(status_code=400, detail="Contact-sales plan cannot be charged via MoMo")

    price_cents = (
        plan.annual_price_cents
        if billing_cycle == BillingCycle.ANNUAL
        else plan.monthly_price_cents
    )
    if not price_cents:
        raise HTTPException(status_code=400, detail="Plan has no price configured")
    amount_ugx = usd_to_ugx(price_cents)

    idempotency_key = f"momo-{organization.id}-{plan_code}-{billing_cycle.value}-{uuid.uuid4()}"

    client = _client()
    result = await client.request_payment(
        phone=phone,
        amount=amount_ugx,
        currency="UGX",
        reason=f"OptiscanAI {plan.display_name} {billing_cycle.value}",
        provider=provider,
    )
    if result.status not in {"pending", "completed"}:
        raise HTTPException(status_code=502, detail=f"MoMo push failed: {result.status}")

    intent = PaymentIntent(
        id=uuid.uuid4(),
        organization_id=organization.id,
        provider=_provider_enum(provider),
        provider_intent_id=result.transaction_id,
        idempotency_key=idempotency_key,
        amount_cents=price_cents,
        currency="USD",
        status=PaymentIntentStatus.PROCESSING,
        phone_msisdn=phone,
        plan_code=plan_code,
        billing_cycle=billing_cycle.value,
        raw_callback={
            "initiated_amount_ugx": amount_ugx,
            "fx_rate": settings.mobile_money.ugx_per_usd,
        },
    )
    db.add(intent)
    await db.flush()
    return intent


# ── Poll status (when no callback arrives) ──


async def poll_momo_status(db: AsyncSession, intent_id: str) -> PaymentIntent:
    intent = await db.get(PaymentIntent, intent_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    if intent.status in {
        PaymentIntentStatus.SUCCEEDED,
        PaymentIntentStatus.FAILED,
        PaymentIntentStatus.CANCELED,
    }:
        return intent

    client = _client()
    provider_str = intent.provider.value
    status = await client.check_payment_status(
        transaction_id=intent.provider_intent_id or "",
        provider=provider_str,
    )
    if status.status == "successful" or status.status == "completed":
        await _confirm_intent(db, intent, raw={"poll": status.message or status.status})
    elif status.status in {"failed", "rejected", "expired"}:
        intent.status = PaymentIntentStatus.FAILED
    return intent


# ── Confirm (callback or successful poll) ──


async def _confirm_intent(
    db: AsyncSession,
    intent: PaymentIntent,
    *,
    raw: dict | None = None,
) -> Subscription:
    """Idempotent: mark intent succeeded, advance Subscription, write Invoice."""
    if intent.status == PaymentIntentStatus.SUCCEEDED:
        # Already processed — fetch existing subscription
        sub = (
            await db.execute(
                select(Subscription).where(Subscription.organization_id == intent.organization_id)
            )
        ).scalar_one()
        return sub

    if raw:
        merged = dict(intent.raw_callback or {})
        merged.update(raw)
        intent.raw_callback = merged

    intent.status = PaymentIntentStatus.SUCCEEDED
    intent.confirmed_at = _utcnow()

    plan = (await db.execute(select(Plan).where(Plan.code == intent.plan_code))).scalar_one()
    cycle = BillingCycle(intent.billing_cycle or "monthly")
    period_start = _utcnow()
    period_end = period_start + _period_delta(cycle)

    sub = (
        await db.execute(
            select(Subscription).where(Subscription.organization_id == intent.organization_id)
        )
    ).scalar_one_or_none()
    if sub is None:
        sub = Subscription(
            id=uuid.uuid4(),
            organization_id=intent.organization_id,
            plan_id=plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=cycle,
            provider=intent.provider,
            current_period_start=period_start,
            current_period_end=period_end,
        )
        db.add(sub)
    else:
        sub.plan_id = plan.id
        sub.status = SubscriptionStatus.ACTIVE
        sub.billing_cycle = cycle
        sub.provider = intent.provider
        sub.current_period_start = period_start
        sub.current_period_end = period_end
        sub.cancel_at_period_end = False
        sub.canceled_at = None
    await db.flush()
    intent.subscription_id = sub.id

    inv = Invoice(
        id=uuid.uuid4(),
        subscription_id=sub.id,
        organization_id=sub.organization_id,
        amount_cents=intent.amount_cents,
        currency="USD",
        status=InvoiceStatus.PAID,
        provider=intent.provider,
        provider_invoice_id=intent.provider_intent_id,
        period_start=period_start,
        period_end=period_end,
        issued_at=_utcnow(),
        paid_at=_utcnow(),
    )
    db.add(inv)
    intent.invoice_id = inv.id

    # No org.plan_id assignment — Organization has no plan_id column;
    # Subscription is the source of truth.
    invalidate_quota_cache(str(sub.organization_id))
    return sub


async def confirm_by_provider_id(
    db: AsyncSession,
    *,
    provider: str,
    provider_intent_id: str,
    raw: dict,
) -> Optional[Subscription]:
    """Find PaymentIntent by provider_intent_id and confirm if not already."""
    provider_enum = _provider_enum(provider)
    intent = (
        await db.execute(
            select(PaymentIntent).where(
                PaymentIntent.provider == provider_enum,
                PaymentIntent.provider_intent_id == provider_intent_id,
            )
        )
    ).scalar_one_or_none()
    if intent is None:
        logger.warning("MoMo callback for unknown intent %s/%s", provider, provider_intent_id)
        return None
    return await _confirm_intent(db, intent, raw=raw)


async def confirm_by_tx_ref(
    db: AsyncSession,
    *,
    provider: str,
    tx_ref: str,
    raw: dict,
) -> Optional[Subscription]:
    """For Flutterwave: idempotency_key carries our reference, tx_ref is set
    in the meta when we initiate.
    """
    provider_enum = _provider_enum(provider)
    intent = (
        await db.execute(
            select(PaymentIntent).where(
                PaymentIntent.provider == provider_enum,
                PaymentIntent.idempotency_key == tx_ref,
            )
        )
    ).scalar_one_or_none()
    if intent is None:
        logger.warning("Flutterwave callback for unknown tx_ref %s", tx_ref)
        return None
    return await _confirm_intent(db, intent, raw=raw)


# ── HMAC signature verification ──


def verify_hmac_signature(body: bytes, presented_sig: str, secret: str) -> bool:
    """Verify a hex-encoded HMAC-SHA256 signature against shared secret."""
    if not secret or not presented_sig:
        return False
    computed = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, presented_sig)


# ── Flutterwave initiate (no PIN push — it's a hosted redirect) ──


async def initiate_flutterwave_payment(
    db: AsyncSession,
    *,
    organization: Organization,
    plan_code: str,
    billing_cycle: BillingCycle,
    customer_email: str,
    customer_name: Optional[str],
    customer_phone: Optional[str],
    user_id: str,
) -> tuple[PaymentIntent, str]:
    if not settings.flutterwave.enabled:
        raise HTTPException(status_code=503, detail="Flutterwave not configured")
    plan = (await db.execute(select(Plan).where(Plan.code == plan_code))).scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Unknown plan {plan_code}")
    if plan.is_contact_sales:
        raise HTTPException(
            status_code=400, detail="Contact-sales plan cannot be charged via Flutterwave"
        )

    price_cents = (
        plan.annual_price_cents
        if billing_cycle == BillingCycle.ANNUAL
        else plan.monthly_price_cents
    )
    if not price_cents:
        raise HTTPException(status_code=400, detail="Plan has no price configured")

    tx_ref = f"flw-{organization.id}-{uuid.uuid4()}"

    intent = PaymentIntent(
        id=uuid.uuid4(),
        organization_id=organization.id,
        provider=PaymentProvider.FLUTTERWAVE,
        provider_intent_id=None,  # filled in from webhook
        idempotency_key=tx_ref,
        amount_cents=price_cents,
        currency="USD",
        status=PaymentIntentStatus.REQUIRES_ACTION,
        plan_code=plan_code,
        billing_cycle=billing_cycle.value,
        raw_callback={},
    )
    db.add(intent)
    await db.flush()

    from backend.app.integrations.flutterwave import client as flw

    redirect_url = (
        f"{settings.public_app_url}/app/checkout/success?provider=flutterwave&tx_ref={tx_ref}"
    )
    body = await flw.initiate_payment(
        tx_ref=tx_ref,
        amount=price_cents / 100.0,
        currency="USD",
        customer_email=customer_email,
        customer_name=customer_name,
        customer_phone=customer_phone,
        redirect_url=redirect_url,
        metadata={
            "organization_id": str(organization.id),
            "plan_code": plan_code,
            "billing_cycle": billing_cycle.value,
            "user_id": user_id,
            "description": f"OptiscanAI {plan.display_name} {billing_cycle.value}",
        },
    )
    payment_link = body.get("data", {}).get("link")
    if not payment_link:
        raise HTTPException(status_code=502, detail="Flutterwave did not return a payment link")
    return intent, payment_link

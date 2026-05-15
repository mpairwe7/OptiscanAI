"""Payments router — Stripe (subscriptions) + MTN MoMo / Airtel Money (referral transport)."""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, get_auth_context
from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.models.membership import MembershipRole
from backend.app.models.subscription import PaymentProvider, Subscription, SubscriptionStatus
from backend.app.models.webhook_event import WebhookEvent
from backend.app.services.billing_service import (
    apply_stripe_invoice_paid,
    apply_stripe_subscription_event,
    get_active_subscription,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


# ────────────────────────────────────────────────────────────────────────────
# Stripe — subscription checkout
# ────────────────────────────────────────────────────────────────────────────


class StripeCheckoutRequest(BaseModel):
    plan_code: str = Field(pattern="^(clinician|practice)$")
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|annual)$")


class StripeCheckoutResponse(BaseModel):
    url: str
    session_id: str


@router.post("/stripe/checkout-session", response_model=StripeCheckoutResponse)
async def create_stripe_checkout(
    body: StripeCheckoutRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> StripeCheckoutResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can subscribe")
    if not settings.stripe.enabled or not settings.stripe.api_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    from backend.app.integrations import stripe_client

    sub = await get_active_subscription(db, str(ctx.organization.id))
    existing_customer_id = sub.stripe_customer_id if sub else None

    idempotency_key = f"checkout-{ctx.organization.id}-{body.plan_code}-{body.billing_cycle}-{uuid.uuid4()}"

    try:
        session = await stripe_client.create_checkout_session(
            plan_code=body.plan_code,
            billing_cycle=body.billing_cycle,
            customer_email=ctx.organization.billing_email or ctx.user.email,
            organization_id=str(ctx.organization.id),
            subscription_id=str(sub.id) if sub else None,
            user_id=str(ctx.user.id),
            idempotency_key=idempotency_key,
            existing_customer_id=existing_customer_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Stripe checkout session creation failed")
        raise HTTPException(status_code=502, detail=f"Stripe error: {exc}")

    if not session.url:
        raise HTTPException(status_code=502, detail="Stripe did not return a checkout URL")
    return StripeCheckoutResponse(url=session.url, session_id=session.id)


@router.post("/stripe/portal-session")
async def create_stripe_portal(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can access the portal")
    sub = await get_active_subscription(db, str(ctx.organization.id))
    if sub is None or not sub.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No active Stripe customer for this org")

    from backend.app.integrations import stripe_client
    try:
        session = await stripe_client.create_portal_session(customer_id=sub.stripe_customer_id)
    except Exception as exc:
        logger.exception("Stripe portal session creation failed")
        raise HTTPException(status_code=502, detail=f"Stripe error: {exc}")
    return {"url": session.url}


# ────────────────────────────────────────────────────────────────────────────
# Stripe webhook
# ────────────────────────────────────────────────────────────────────────────


async def _is_already_processed(db: AsyncSession, provider: PaymentProvider, event_id: str) -> bool:
    """Idempotency: record-or-skip in webhook_events."""
    existing = (
        await db.execute(
            select(WebhookEvent).where(
                WebhookEvent.provider == provider,
                WebhookEvent.provider_event_id == event_id,
            )
        )
    ).scalar_one_or_none()
    return existing is not None and existing.processed_at is not None


async def _record_webhook(
    db: AsyncSession,
    *,
    provider: PaymentProvider,
    event_id: str,
    event_type: str,
    payload: dict,
    error: Optional[str] = None,
) -> WebhookEvent:
    """Insert-or-update a webhook_events row.

    Stores the full raw provider payload so an operator can replay later. The
    unique constraint on ``(provider, provider_event_id)`` makes this safe to
    call from any retry path.
    """
    from datetime import datetime, timezone

    existing = (
        await db.execute(
            select(WebhookEvent).where(
                WebhookEvent.provider == provider,
                WebhookEvent.provider_event_id == event_id,
            )
        )
    ).scalar_one_or_none()
    now_ = datetime.now(timezone.utc)
    if existing is not None:
        if not error:
            existing.processed_at = now_
            existing.error = None
        else:
            existing.error = error
        if payload:
            existing.payload = payload
        return existing
    ev = WebhookEvent(
        id=uuid.uuid4(),
        provider=provider,
        provider_event_id=event_id,
        event_type=event_type,
        payload=payload,
        processed_at=now_ if not error else None,
        error=error,
    )
    db.add(ev)
    return ev


@router.post("/stripe/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(default="", alias="stripe-signature"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if not settings.stripe.enabled or not settings.stripe.webhook_secret:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    from backend.app.integrations import stripe_client

    raw = await request.body()
    try:
        event = stripe_client.verify_webhook(raw, stripe_signature)
    except Exception as exc:
        logger.warning("Stripe webhook signature verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event["id"]
    event_type = event["type"]

    if await _is_already_processed(db, PaymentProvider.STRIPE, event_id):
        return {"status": "duplicate", "event_id": event_id}

    obj = event["data"]["object"]
    try:
        if event_type in {
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        }:
            await apply_stripe_subscription_event(db, stripe_subscription=obj)
        elif event_type == "checkout.session.completed":
            sub_id = obj.get("subscription")
            if sub_id:
                stripe_sub = await stripe_client.retrieve_subscription(sub_id)
                # Merge checkout metadata into the subscription metadata for plan_code/billing_cycle
                checkout_meta = obj.get("metadata") or {}
                sub_meta = dict(stripe_sub.get("metadata") or {})
                for key in ("plan_code", "billing_cycle", "organization_id"):
                    sub_meta.setdefault(key, checkout_meta.get(key, ""))
                stripe_sub_dict = dict(stripe_sub)
                stripe_sub_dict["metadata"] = sub_meta
                await apply_stripe_subscription_event(db, stripe_subscription=stripe_sub_dict)
        elif event_type == "invoice.paid":
            await apply_stripe_invoice_paid(db, stripe_invoice=obj)
        elif event_type == "invoice.payment_failed":
            sub_id = obj.get("subscription")
            if sub_id:
                row = (
                    await db.execute(
                        select(Subscription).where(Subscription.stripe_subscription_id == sub_id)
                    )
                ).scalar_one_or_none()
                if row is not None:
                    row.status = SubscriptionStatus.PAST_DUE
        else:
            logger.debug("Unhandled Stripe event type: %s", event_type)
    except Exception as exc:
        logger.exception("Stripe webhook handler failed for event %s", event_id)
        await _record_webhook(
            db,
            provider=PaymentProvider.STRIPE,
            event_id=event_id,
            event_type=event_type,
            payload=dict(event),
            error=str(exc)[:2000],
        )
        # Return 500 so Stripe retries
        raise HTTPException(status_code=500, detail="Webhook processing failed")

    await _record_webhook(
        db,
        provider=PaymentProvider.STRIPE,
        event_id=event_id,
        event_type=event_type,
        payload=dict(event),
    )
    return {"status": "ok", "event_id": event_id, "event_type": event_type}


# ────────────────────────────────────────────────────────────────────────────
# MoMo / Flutterwave — subscription checkout
# ────────────────────────────────────────────────────────────────────────────


class MoMoCheckoutRequest(BaseModel):
    plan_code: str = Field(pattern="^(clinician|practice)$")
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|annual)$")
    phone: str = Field(min_length=10, max_length=20)
    provider: str = Field(pattern="^(mtn|airtel)$")


class MoMoCheckoutResponse(BaseModel):
    intent_id: str
    status: str
    provider: str
    poll_url: str


@router.post("/momo/checkout", response_model=MoMoCheckoutResponse)
async def momo_checkout(
    body: MoMoCheckoutRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> MoMoCheckoutResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can subscribe")

    from backend.app.models.subscription import BillingCycle
    from backend.app.services.momo_billing_service import initiate_momo_payment

    cycle = BillingCycle(body.billing_cycle)
    intent = await initiate_momo_payment(
        db,
        organization=ctx.organization,
        plan_code=body.plan_code,
        billing_cycle=cycle,
        phone=body.phone,
        provider=body.provider,
        user_id=str(ctx.user.id),
    )
    return MoMoCheckoutResponse(
        intent_id=str(intent.id),
        status=intent.status.value,
        provider=intent.provider.value,
        poll_url=f"/api/v1/payments/intents/{intent.id}",
    )


class FlutterwaveCheckoutRequest(BaseModel):
    plan_code: str = Field(pattern="^(clinician|practice)$")
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|annual)$")
    phone: str = Field(default="", max_length=20)


class FlutterwaveCheckoutResponse(BaseModel):
    intent_id: str
    payment_url: str


@router.post("/flutterwave/checkout", response_model=FlutterwaveCheckoutResponse)
async def flutterwave_checkout(
    body: FlutterwaveCheckoutRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> FlutterwaveCheckoutResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can subscribe")

    from backend.app.models.subscription import BillingCycle
    from backend.app.services.momo_billing_service import initiate_flutterwave_payment

    cycle = BillingCycle(body.billing_cycle)
    intent, payment_link = await initiate_flutterwave_payment(
        db,
        organization=ctx.organization,
        plan_code=body.plan_code,
        billing_cycle=cycle,
        customer_email=ctx.organization.billing_email or ctx.user.email,
        customer_name=ctx.user.full_name,
        customer_phone=body.phone or None,
        user_id=str(ctx.user.id),
    )
    return FlutterwaveCheckoutResponse(intent_id=str(intent.id), payment_url=payment_link)


@router.get("/intents/{intent_id}")
async def get_payment_intent(
    intent_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from backend.app.models.payment_intent import PaymentIntent
    from backend.app.services.momo_billing_service import poll_momo_status

    intent = await db.get(PaymentIntent, intent_id)
    if intent is None or str(intent.organization_id) != str(ctx.organization.id):
        raise HTTPException(status_code=404, detail="Intent not found")

    # For MoMo, poll the provider for live status; Flutterwave waits for webhook
    if intent.provider in {PaymentProvider.MTN, PaymentProvider.AIRTEL}:
        intent = await poll_momo_status(db, str(intent.id))

    return {
        "id": str(intent.id),
        "status": intent.status.value,
        "provider": intent.provider.value,
        "plan_code": intent.plan_code,
        "billing_cycle": intent.billing_cycle,
        "amount_cents": intent.amount_cents,
        "currency": intent.currency,
        "confirmed_at": intent.confirmed_at.isoformat() if intent.confirmed_at else None,
    }


# ── Public webhooks (no auth, signature-verified) ──


@router.post("/momo/callback/mtn", include_in_schema=False)
async def mtn_callback(
    request: Request,
    x_callback_signature: str = Header(default="", alias="x-callback-signature"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from backend.app.services.momo_billing_service import (
        confirm_by_provider_id,
        verify_hmac_signature,
    )

    body_bytes = await request.body()
    if settings.mobile_money.mtn_callback_secret and not verify_hmac_signature(
        body_bytes, x_callback_signature, settings.mobile_money.mtn_callback_secret,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    import json
    try:
        payload = json.loads(body_bytes.decode())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    tx_id = (
        payload.get("referenceId")
        or payload.get("externalId")
        or payload.get("transaction_id")
        or ""
    )
    status_str = (payload.get("status") or "").lower()

    # Idempotency: one webhook_events row per (provider, externalId+status)
    if not tx_id:
        return {"status": "ignored", "reason": "missing reference"}

    event_id = f"mtn:{tx_id}:{status_str}"
    if await _is_already_processed(db, PaymentProvider.MTN, event_id):
        return {"status": "duplicate"}

    if status_str in {"successful", "completed"}:
        await confirm_by_provider_id(
            db, provider="mtn", provider_intent_id=tx_id, raw=payload,
        )

    await _record_webhook(
        db,
        provider=PaymentProvider.MTN,
        event_id=event_id,
        event_type=f"mtn.{status_str}",
        payload=payload,
    )
    return {"status": "ok"}


@router.post("/momo/callback/airtel", include_in_schema=False)
async def airtel_callback(
    request: Request,
    x_signature: str = Header(default="", alias="x-signature"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from backend.app.services.momo_billing_service import (
        confirm_by_provider_id,
        verify_hmac_signature,
    )

    body_bytes = await request.body()
    if settings.mobile_money.airtel_callback_secret and not verify_hmac_signature(
        body_bytes, x_signature, settings.mobile_money.airtel_callback_secret,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    import json
    try:
        payload = json.loads(body_bytes.decode())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    tx_id = (
        payload.get("transaction", {}).get("id")
        or payload.get("transaction_id")
        or payload.get("reference")
        or ""
    )
    status_str = (
        payload.get("transaction", {}).get("status_code")
        or payload.get("status", "")
    ).lower()
    if not tx_id:
        return {"status": "ignored", "reason": "missing reference"}

    event_id = f"airtel:{tx_id}:{status_str}"
    if await _is_already_processed(db, PaymentProvider.AIRTEL, event_id):
        return {"status": "duplicate"}

    if status_str in {"success", "ts", "successful", "completed"}:
        await confirm_by_provider_id(
            db, provider="airtel", provider_intent_id=tx_id, raw=payload,
        )

    await _record_webhook(
        db,
        provider=PaymentProvider.AIRTEL,
        event_id=event_id,
        event_type=f"airtel.{status_str}",
        payload=payload,
    )
    return {"status": "ok"}


@router.post("/flutterwave/webhook", include_in_schema=False)
async def flutterwave_webhook(
    request: Request,
    verif_hash: str = Header(default="", alias="verif-hash"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from backend.app.integrations.flutterwave import client as flw
    from backend.app.services.momo_billing_service import confirm_by_tx_ref

    if not flw.verify_webhook_hash(verif_hash):
        raise HTTPException(status_code=401, detail="Invalid verif-hash")

    body = await request.json()
    event_type = body.get("event") or body.get("type") or "transaction"
    data = body.get("data") or body
    tx_ref = data.get("tx_ref") or data.get("txRef") or ""
    fw_status = (data.get("status") or "").lower()
    tx_id_remote = data.get("id") or data.get("flw_ref") or ""

    if not tx_ref:
        return {"status": "ignored", "reason": "missing tx_ref"}

    event_id = f"flutterwave:{tx_ref}:{fw_status}"
    if await _is_already_processed(db, PaymentProvider.FLUTTERWAVE, event_id):
        return {"status": "duplicate"}

    if fw_status in {"successful", "success", "completed"}:
        # Double-check via Flutterwave API
        try:
            verified = await flw.verify_transaction_by_ref(tx_ref)
            verified_status = verified.get("data", {}).get("status", "").lower()
            if verified_status not in {"successful", "success"}:
                logger.warning("Flutterwave webhook claims success but verify says %s", verified_status)
                return {"status": "rejected", "reason": "verify mismatch"}
        except Exception as exc:
            logger.warning("Flutterwave verify failed: %s — trusting webhook", exc)
        await confirm_by_tx_ref(db, provider="flutterwave", tx_ref=tx_ref, raw=data)

    await _record_webhook(
        db,
        provider=PaymentProvider.FLUTTERWAVE,
        event_id=event_id,
        event_type=f"flutterwave.{event_type}.{fw_status}",
        payload=body,
    )
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────────────────────
# Mobile money — referral transport (Phase 3 feature, separate from subscriptions)
# ────────────────────────────────────────────────────────────────────────────


class PaymentRequestBody(BaseModel):
    phone: str = Field(..., min_length=10)
    amount: int = Field(default=50000, ge=1000)
    currency: str = "UGX"
    reason: str = "Referral transport support"
    provider: str = "auto"


@router.post("/request")
async def request_payment(body: PaymentRequestBody):
    """Initiate referral transport payment to patient's phone."""
    from backend.app.integrations.mobile_money.client import MobileMoneyClient

    mm = getattr(settings, "mobile_money", None)
    if not mm or not mm.enabled:
        raise HTTPException(status_code=404, detail="Mobile money not enabled")

    client = MobileMoneyClient(
        mtn_api_key=mm.mtn_api_key,
        mtn_subscription_key=mm.mtn_subscription_key,
        mtn_environment=mm.mtn_environment,
        airtel_client_id=mm.airtel_client_id,
    )
    result = await client.request_payment(
        phone=body.phone, amount=body.amount,
        currency=body.currency, reason=body.reason, provider=body.provider,
    )
    return {"transaction_id": result.transaction_id, "status": result.status, "provider": result.provider}


@router.get("/status/{tx_id}")
async def payment_status(tx_id: str, provider: str = "mtn"):
    """Check payment status."""
    from backend.app.integrations.mobile_money.client import MobileMoneyClient
    client = MobileMoneyClient()
    status_ = await client.check_payment_status(tx_id, provider)
    return {"transaction_id": status_.transaction_id, "status": status_.status}


@router.post("/callback/mtn")
async def mtn_callback(body: dict):
    """MTN MoMo webhook callback (signature verification TODO Phase E)."""
    logger.info("MTN callback: %s", body)
    return {"status": "received"}


@router.post("/callback/airtel")
async def airtel_callback(body: dict):
    """Airtel Money webhook callback (signature verification TODO Phase E)."""
    logger.info("Airtel callback: %s", body)
    return {"status": "received"}

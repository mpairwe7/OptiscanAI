"""Payments router — MTN MoMo subscription checkout + referral transport."""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, get_auth_context
from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.models.membership import MembershipRole
from backend.app.models.subscription import PaymentProvider
from backend.app.models.webhook_event import WebhookEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


# ────────────────────────────────────────────────────────────────────────────
# Webhook idempotency helpers (MTN callback uses these)
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


# ────────────────────────────────────────────────────────────────────────────
# MTN MoMo — subscription checkout
# ────────────────────────────────────────────────────────────────────────────


class MoMoCheckoutRequest(BaseModel):
    plan_code: str = Field(pattern="^(clinician|practice)$")
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|annual)$")
    phone: str = Field(min_length=10, max_length=20)


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
        provider="mtn",
        user_id=str(ctx.user.id),
    )
    return MoMoCheckoutResponse(
        intent_id=str(intent.id),
        status=intent.status.value,
        provider=intent.provider.value,
        poll_url=f"/api/v1/payments/intents/{intent.id}",
    )


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

    if intent.provider == PaymentProvider.MTN:
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


# ── Public webhook (no auth, signature-verified) ──


@router.post("/momo/callback/mtn", include_in_schema=False)
async def mtn_subscription_callback(
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
async def mtn_transport_callback(body: dict):
    """MTN MoMo referral-transport webhook (signature verification TODO Phase E)."""
    logger.info("MTN transport callback: %s", body)
    return {"status": "received"}

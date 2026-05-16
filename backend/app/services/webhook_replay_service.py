"""Webhook replay — re-process a stored WebhookEvent payload.

Used by ops via the admin endpoints to recover from a transient handler
failure or to manually re-sync after a downstream code fix.

The provider-specific replay logic mirrors the live webhook handlers in
``backend.app.routers.payments`` — kept here so it can be invoked without
HTTP middleware.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.subscription import (
    PaymentProvider,
    Subscription,
    SubscriptionStatus,
)
from backend.app.models.webhook_event import WebhookEvent
from backend.app.services.billing_service import (
    apply_stripe_invoice_paid,
    apply_stripe_subscription_event,
)
from backend.app.services.momo_billing_service import (
    confirm_by_provider_id,
    confirm_by_tx_ref,
)

logger = logging.getLogger(__name__)


async def replay_webhook(db: AsyncSession, event_id: str) -> dict:
    """Re-process the stored webhook payload. Returns a result dict for the
    operator UI; never raises (errors land in ``WebhookEvent.error`` + result).
    """
    event = await db.get(WebhookEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Webhook event not found")

    payload = event.payload or {}
    if not payload:
        return {
            "status": "no_payload",
            "message": "This event was stored without its raw payload — pre-replay-feature row.",
        }

    provider = event.provider
    try:
        if provider == PaymentProvider.STRIPE:
            result = await _replay_stripe(db, payload)
        elif provider == PaymentProvider.MTN:
            result = await _replay_mtn(db, payload)
        elif provider == PaymentProvider.AIRTEL:
            result = await _replay_airtel(db, payload)
        elif provider == PaymentProvider.FLUTTERWAVE:
            result = await _replay_flutterwave(db, payload)
        else:
            return {"status": "unsupported_provider", "provider": provider.value}
    except Exception as exc:
        logger.exception("Replay failed for event %s", event_id)
        event.error = str(exc)[:2000]
        return {"status": "error", "error": str(exc)}

    event.processed_at = datetime.now(timezone.utc)
    event.error = None
    return {"status": "replayed", "provider": provider.value, **result}


# ── Per-provider replay impls ──


async def _replay_stripe(db: AsyncSession, event: dict) -> dict:
    event_type = event.get("type") or ""
    obj = (event.get("data") or {}).get("object") or {}
    summary: dict[str, Optional[str]] = {"event_type": event_type, "object_id": obj.get("id")}

    if event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        sub = await apply_stripe_subscription_event(db, stripe_subscription=obj)
        summary["subscription_id"] = str(sub.id) if sub else None
    elif event_type == "checkout.session.completed":
        sub_id = obj.get("subscription")
        if sub_id:
            from backend.app.integrations import stripe_client

            stripe_sub = await stripe_client.retrieve_subscription(sub_id)
            checkout_meta = obj.get("metadata") or {}
            sub_meta = dict(stripe_sub.get("metadata") or {})
            for key in ("plan_code", "billing_cycle", "organization_id"):
                sub_meta.setdefault(key, checkout_meta.get(key, ""))
            stripe_sub_dict = dict(stripe_sub)
            stripe_sub_dict["metadata"] = sub_meta
            sub = await apply_stripe_subscription_event(db, stripe_subscription=stripe_sub_dict)
            summary["subscription_id"] = str(sub.id) if sub else None
    elif event_type == "invoice.paid":
        inv = await apply_stripe_invoice_paid(db, stripe_invoice=obj)
        summary["invoice_id"] = str(inv.id) if inv else None
    elif event_type == "invoice.payment_failed":
        from sqlalchemy import select

        sub_id = obj.get("subscription")
        if sub_id:
            row = (
                await db.execute(
                    select(Subscription).where(Subscription.stripe_subscription_id == sub_id)
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = SubscriptionStatus.PAST_DUE
                summary["flipped_to_past_due"] = str(row.id)
    return summary


async def _replay_mtn(db: AsyncSession, payload: dict) -> dict:
    tx_id = (
        payload.get("referenceId")
        or payload.get("externalId")
        or payload.get("transaction_id")
        or ""
    )
    status_str = (payload.get("status") or "").lower()
    if not tx_id:
        return {"action": "ignored", "reason": "missing reference"}
    if status_str in {"successful", "completed"}:
        sub = await confirm_by_provider_id(
            db,
            provider="mtn",
            provider_intent_id=tx_id,
            raw=payload,
        )
        return {"action": "confirmed", "subscription_id": str(sub.id) if sub else None}
    return {"action": "noop", "status": status_str}


async def _replay_airtel(db: AsyncSession, payload: dict) -> dict:
    tx = payload.get("transaction") or {}
    tx_id = tx.get("id") or payload.get("transaction_id") or payload.get("reference") or ""
    status_str = (tx.get("status_code") or payload.get("status") or "").lower()
    if not tx_id:
        return {"action": "ignored", "reason": "missing reference"}
    if status_str in {"success", "ts", "successful", "completed"}:
        sub = await confirm_by_provider_id(
            db,
            provider="airtel",
            provider_intent_id=tx_id,
            raw=payload,
        )
        return {"action": "confirmed", "subscription_id": str(sub.id) if sub else None}
    return {"action": "noop", "status": status_str}


async def _replay_flutterwave(db: AsyncSession, body: dict) -> dict:
    data = body.get("data") or body
    tx_ref = data.get("tx_ref") or data.get("txRef") or ""
    fw_status = (data.get("status") or "").lower()
    if not tx_ref:
        return {"action": "ignored", "reason": "missing tx_ref"}
    if fw_status in {"successful", "success", "completed"}:
        sub = await confirm_by_tx_ref(
            db,
            provider="flutterwave",
            tx_ref=tx_ref,
            raw=data,
        )
        return {"action": "confirmed", "subscription_id": str(sub.id) if sub else None}
    return {"action": "noop", "status": fw_status}

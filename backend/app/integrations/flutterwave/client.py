"""Flutterwave Standard checkout client.

Flutterwave's "Standard" flow:
  1. POST /v3/payments with tx_ref + customer + amount + redirect_url + meta
  2. Receive { status, data: { link } } — redirect the browser to `link`
  3. User completes payment on Flutterwave's hosted page
  4. Flutterwave fires a webhook to your endpoint with verif-hash header
  5. You verify the hash and the transaction (GET /v3/transactions/verify_by_reference)
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Optional

import httpx

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    if not settings.flutterwave.secret_key:
        raise RuntimeError("Flutterwave not configured — set FLUTTERWAVE__SECRET_KEY")
    return {
        "Authorization": f"Bearer {settings.flutterwave.secret_key}",
        "Content-Type": "application/json",
    }


async def initiate_payment(
    *,
    tx_ref: str,
    amount: float,
    currency: str,
    customer_email: str,
    customer_name: Optional[str],
    customer_phone: Optional[str],
    redirect_url: str,
    metadata: dict[str, Any],
    payment_options: str = "card,mobilemoneyuganda,mobilemoneyzambia,mobilemoneyrwanda,mobilemoneyghana",
) -> dict[str, Any]:
    payload = {
        "tx_ref": tx_ref,
        "amount": amount,
        "currency": currency,
        "redirect_url": redirect_url,
        "customer": {
            "email": customer_email,
            "name": customer_name or customer_email,
            "phonenumber": customer_phone or "",
        },
        "customizations": {
            "title": "OptiscanAI subscription",
            "description": metadata.get("description", "OptiscanAI plan upgrade"),
        },
        "meta": metadata,
        "payment_options": payment_options,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{settings.flutterwave.base_url}/payments",
            headers=_headers(),
            json=payload,
        )
        r.raise_for_status()
        body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Flutterwave init failed: {body}")
    return body


async def verify_transaction_by_ref(tx_ref: str) -> dict[str, Any]:
    """After webhook, double-check the transaction is real."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{settings.flutterwave.base_url}/transactions/verify_by_reference",
            headers=_headers(),
            params={"tx_ref": tx_ref},
        )
        r.raise_for_status()
        body = r.json()
    return body


def verify_webhook_hash(presented: str) -> bool:
    """Flutterwave webhook auth: compare verif-hash header to configured secret.

    Constant-time comparison to avoid timing attacks.
    """
    expected = settings.flutterwave.secret_hash
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected, presented)

"""Mobile money payments router (MTN MoMo / Airtel Money)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["mobile-money"])


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
    from backend.app.core.config import settings

    mm = getattr(settings, "mobile_money", None)
    if not mm or not mm.enabled:
        raise HTTPException(404, "Mobile money not enabled")

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
    status = await client.check_payment_status(tx_id, provider)
    return {"transaction_id": status.transaction_id, "status": status.status}


@router.post("/callback/mtn")
async def mtn_callback(body: dict):
    """MTN MoMo webhook callback."""
    logger.info("MTN callback: %s", body)
    return {"status": "received"}


@router.post("/callback/airtel")
async def airtel_callback(body: dict):
    """Airtel Money webhook callback."""
    logger.info("Airtel callback: %s", body)
    return {"status": "received"}

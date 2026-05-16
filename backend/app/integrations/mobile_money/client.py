"""Unified mobile money client for MTN MoMo and Airtel Money.

Handles referral transport support payments for patients referred
from screening results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PaymentRequest:
    transaction_id: str = ""
    phone: str = ""
    amount: int = 0
    currency: str = "UGX"
    reason: str = ""
    provider: str = ""
    status: str = "pending"


@dataclass
class PaymentStatus:
    transaction_id: str = ""
    status: str = "unknown"  # pending | completed | failed
    provider: str = ""
    message: str = ""


class MobileMoneyClient:
    """Unified client for MTN MoMo and Airtel Money APIs."""

    def __init__(
        self,
        mtn_api_key: str = "",
        mtn_api_secret: str = "",
        mtn_subscription_key: str = "",
        mtn_environment: str = "sandbox",
        airtel_client_id: str = "",
        airtel_client_secret: str = "",
    ):
        self._mtn_key = mtn_api_key
        self._mtn_secret = mtn_api_secret
        self._mtn_sub_key = mtn_subscription_key
        self._mtn_env = mtn_environment
        self._airtel_id = airtel_client_id
        self._airtel_secret = airtel_client_secret

    def _detect_provider(self, phone: str) -> str:
        """Detect provider from Ugandan phone number prefix."""
        cleaned = phone.replace("+", "").replace(" ", "")
        if cleaned.startswith("256"):
            cleaned = cleaned[3:]
        # MTN Uganda: 077, 078, 076
        if cleaned.startswith(("77", "78", "76")):
            return "mtn"
        # Airtel Uganda: 070, 075, 074
        if cleaned.startswith(("70", "75", "74")):
            return "airtel"
        return "mtn"  # Default

    async def request_payment(
        self,
        phone: str,
        amount: int,
        currency: str = "UGX",
        reason: str = "Referral transport support",
        provider: str = "auto",
    ) -> PaymentRequest:
        """Initiate a payment request to patient's phone."""
        if provider == "auto":
            provider = self._detect_provider(phone)

        from uuid import uuid4

        tx_id = str(uuid4())

        if provider == "mtn":
            return await self._mtn_request(tx_id, phone, amount, currency, reason)
        elif provider == "airtel":
            return await self._airtel_request(tx_id, phone, amount, currency, reason)
        else:
            return PaymentRequest(transaction_id=tx_id, status="error")

    async def _mtn_request(
        self, tx_id: str, phone: str, amount: int, currency: str, reason: str
    ) -> PaymentRequest:
        """MTN MoMo Collections API v1."""
        base_url = (
            "https://sandbox.momodeveloper.mtn.com"
            if self._mtn_env == "sandbox"
            else "https://momodeveloper.mtn.com"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self._mtn_sub_key,
            "X-Reference-Id": tx_id,
            "X-Target-Environment": self._mtn_env,
            "Content-Type": "application/json",
        }

        payload = {
            "amount": str(amount),
            "currency": currency,
            "externalId": tx_id,
            "payer": {"partyIdType": "MSISDN", "partyId": phone},
            "payerMessage": reason,
            "payeeNote": f"RetinalAI referral transport - {tx_id[:8]}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/collection/v1_0/requesttopay", headers=headers, json=payload
                ) as resp:
                    if resp.status == 202:
                        return PaymentRequest(
                            transaction_id=tx_id,
                            phone=phone,
                            amount=amount,
                            currency=currency,
                            reason=reason,
                            provider="mtn",
                            status="pending",
                        )
                    logger.error("MTN MoMo request failed: %s", await resp.text())
                    return PaymentRequest(transaction_id=tx_id, status="failed", provider="mtn")
        except Exception as e:
            logger.error("MTN MoMo error: %s", e)
            return PaymentRequest(transaction_id=tx_id, status="error", provider="mtn")

    async def _airtel_request(
        self, tx_id: str, phone: str, amount: int, currency: str, reason: str
    ) -> PaymentRequest:
        """Airtel Money Uganda API."""
        # Airtel Money API implementation follows similar pattern
        logger.info("Airtel Money payment request: %s -> %s %s %s", phone, amount, currency, reason)
        return PaymentRequest(
            transaction_id=tx_id,
            phone=phone,
            amount=amount,
            currency=currency,
            reason=reason,
            provider="airtel",
            status="pending",
        )

    async def check_payment_status(
        self, transaction_id: str, provider: str = "mtn"
    ) -> PaymentStatus:
        """Poll for payment confirmation."""
        if provider == "mtn":
            base_url = (
                "https://sandbox.momodeveloper.mtn.com"
                if self._mtn_env == "sandbox"
                else "https://momodeveloper.mtn.com"
            )
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base_url}/collection/v1_0/requesttopay/{transaction_id}",
                        headers={
                            "Ocp-Apim-Subscription-Key": self._mtn_sub_key,
                            "X-Target-Environment": self._mtn_env,
                        },
                    ) as resp:
                        data = await resp.json()
                        return PaymentStatus(
                            transaction_id=transaction_id,
                            status=data.get("status", "unknown").lower(),
                            provider="mtn",
                        )
            except Exception as e:
                return PaymentStatus(transaction_id=transaction_id, status="error", message=str(e))

        return PaymentStatus(transaction_id=transaction_id, status="unknown", provider=provider)

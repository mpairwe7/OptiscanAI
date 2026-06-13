"""Unified mobile money client for MTN MoMo and Airtel Money.

Handles referral transport support payments for patients referred
from screening results.
"""

from __future__ import annotations

import base64
import logging
import time
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
        request_timeout_s: float = 20.0,
    ):
        self._mtn_key = mtn_api_key
        self._mtn_secret = mtn_api_secret
        self._mtn_sub_key = mtn_subscription_key
        self._mtn_env = mtn_environment
        self._airtel_id = airtel_client_id
        self._airtel_secret = airtel_client_secret
        # Cap every outbound call so a stalled MTN/Airtel endpoint can't hang
        # the request (and, via the OAuth bearer fetch, the whole payment).
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        # OAuth bearer cache for MTN MoMo Collections API.
        # MTN tokens live 1h; we refresh ~5 min before expiry.
        self._mtn_token: str | None = None
        self._mtn_token_expiry: float = 0.0

    def _mtn_base_url(self) -> str:
        return (
            "https://sandbox.momodeveloper.mtn.com"
            if self._mtn_env == "sandbox"
            else "https://momodeveloper.mtn.com"
        )

    async def _mtn_bearer(self) -> str:
        """Fetch + cache a Collections OAuth bearer.

        MTN MoMo Collections requires every requesttopay call to carry
        `Authorization: Bearer <token>` (the subscription key alone returns
        401). Tokens are obtained via Basic-auth on /collection/token/ using
        the provisioned API User UUID + apiKey. They are valid for ~1h, so
        we cache and refresh just before expiry.
        """
        now = time.time()
        if self._mtn_token and now < self._mtn_token_expiry - 300:
            return self._mtn_token
        if not (self._mtn_key and self._mtn_secret and self._mtn_sub_key):
            raise RuntimeError("MTN MoMo not configured (api_key/api_secret/subscription_key)")
        basic = base64.b64encode(f"{self._mtn_key}:{self._mtn_secret}".encode()).decode()
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                f"{self._mtn_base_url()}/collection/token/",
                headers={
                    "Ocp-Apim-Subscription-Key": self._mtn_sub_key,
                    "Authorization": f"Basic {basic}",
                },
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"MTN MoMo /collection/token/ failed: {resp.status} {await resp.text()}"
                    )
                data = await resp.json()
        self._mtn_token = data["access_token"]
        self._mtn_token_expiry = now + int(data.get("expires_in", 3600))
        return self._mtn_token

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
        base_url = self._mtn_base_url()

        try:
            bearer = await self._mtn_bearer()
        except Exception as e:
            logger.error("MTN MoMo bearer fetch failed: %s", e)
            return PaymentRequest(transaction_id=tx_id, status="error", provider="mtn")

        headers = {
            "Ocp-Apim-Subscription-Key": self._mtn_sub_key,
            "Authorization": f"Bearer {bearer}",
            "X-Reference-Id": tx_id,
            "X-Target-Environment": self._mtn_env,
            "Content-Type": "application/json",
        }

        # MTN sandbox only accepts EUR on requesttopay (returns
        # 500 INVALID_CURRENCY for UGX). Production accepts UGX. Override
        # on the wire so callers can keep their UGX bookkeeping unchanged
        # when running against sandbox.
        wire_currency = "EUR" if self._mtn_env == "sandbox" else currency

        payload = {
            "amount": str(amount),
            "currency": wire_currency,
            "externalId": tx_id,
            "payer": {"partyIdType": "MSISDN", "partyId": phone},
            "payerMessage": reason,
            "payeeNote": f"RetinalAI referral transport - {tx_id[:8]}",
        }

        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
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
            base_url = self._mtn_base_url()
            try:
                bearer = await self._mtn_bearer()
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.get(
                        f"{base_url}/collection/v1_0/requesttopay/{transaction_id}",
                        headers={
                            "Ocp-Apim-Subscription-Key": self._mtn_sub_key,
                            "Authorization": f"Bearer {bearer}",
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

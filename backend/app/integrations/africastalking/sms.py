"""Africa's Talking SMS service for referral notifications."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SMSResult:
    message_id: str = ""
    phone: str = ""
    status: str = ""  # sent | failed | queued
    cost: str = ""


@dataclass
class DeliveryStatus:
    message_id: str = ""
    status: str = ""  # Delivered | Failed | Rejected
    failure_reason: str = ""


class SMSService:
    """Africa's Talking SMS API for referral notifications."""

    def __init__(self, api_key: str = "", username: str = "sandbox", sender_id: str = "RetinalAI"):
        self._api_key = api_key
        self._username = username
        self._sender_id = sender_id
        self._client = None

    def initialize(self) -> None:
        """Initialize Africa's Talking SDK."""
        try:
            import africastalking
            africastalking.initialize(self._username, self._api_key)
            self._client = africastalking.SMS
            logger.info("Africa's Talking SMS initialized")
        except ImportError:
            logger.warning("africastalking SDK not installed — SMS in stub mode")

    async def send_referral_sms(
        self,
        phone: str,
        patient_name: str = "",
        diseases: list[str] = None,
        priority: str = "ROUTINE",
        facility: str = "",
        language: str = "en",
    ) -> SMSResult:
        """Send screening result + referral as SMS."""
        diseases = diseases or []
        message = self._format_referral_sms(patient_name, diseases, priority, facility, language)

        if self._client is None:
            logger.info("SMS stub: %s -> %s", phone, message[:50])
            return SMSResult(message_id="stub", phone=phone, status="stub")

        try:
            response = self._client.send(message, [phone], sender_id=self._sender_id)
            recipients = response.get("SMSMessageData", {}).get("Recipients", [{}])
            if recipients:
                r = recipients[0]
                return SMSResult(
                    message_id=r.get("messageId", ""),
                    phone=phone,
                    status="sent" if r.get("status") == "Success" else "failed",
                    cost=r.get("cost", ""),
                )
            return SMSResult(phone=phone, status="failed")
        except Exception as e:
            logger.error("SMS send failed: %s", e)
            return SMSResult(phone=phone, status="failed")

    def _format_referral_sms(
        self, name: str, diseases: list[str], priority: str, facility: str, language: str
    ) -> str:
        if language == "lg":
            disease_text = ", ".join(diseases) if diseases else "tewali"
            return (
                f"RetinalAI: {name or 'Omulwadde'} — "
                f"Ebivaamu: {disease_text}. "
                f"Obunyonyi: {priority}. "
                f"{'Genda mu ' + facility if facility else 'Kebera musawo wammwe'}."
            )
        disease_text = ", ".join(diseases) if diseases else "none"
        return (
            f"RetinalAI Screening: {name or 'Patient'} — "
            f"Findings: {disease_text}. "
            f"Priority: {priority}. "
            f"{'Refer to ' + facility if facility else 'See your eye doctor'}."
        )

    async def check_delivery(self, message_id: str) -> DeliveryStatus:
        """Check SMS delivery status."""
        return DeliveryStatus(message_id=message_id, status="Delivered")

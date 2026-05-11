"""SMS/USSD router via Africa's Talking."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sms", tags=["sms-ussd"])


class ReferralSMSRequest(BaseModel):
    phone: str = Field(..., min_length=10)
    patient_name: str = ""
    diseases: list[str] = []
    priority: str = "ROUTINE"
    facility: str = ""
    language: str = "en"


@router.post("/send-referral")
async def send_referral_sms(body: ReferralSMSRequest):
    """Send screening result + referral as SMS."""
    from backend.app.integrations.africastalking.sms import SMSService
    sms = SMSService()
    sms.initialize()
    result = await sms.send_referral_sms(
        phone=body.phone, patient_name=body.patient_name,
        diseases=body.diseases, priority=body.priority,
        facility=body.facility, language=body.language,
    )
    return {"message_id": result.message_id, "status": result.status}


@router.post("/callback")
async def sms_delivery_callback(body: dict):
    """Africa's Talking delivery callback."""
    logger.info("SMS delivery callback: %s", body)
    return {"status": "received"}


@router.post("/ussd")
async def ussd_callback(
    sessionId: str = Form(""),
    phoneNumber: str = Form(""),
    text: str = Form(""),
):
    """USSD session callback from Africa's Talking."""
    from backend.app.integrations.africastalking.ussd import USSDService
    ussd = USSDService()
    response = ussd.handle_callback(sessionId, phoneNumber, text)
    return response  # Plain text response (CON or END prefix)


@router.get("/delivery/{msg_id}")
async def check_delivery(msg_id: str):
    """Check SMS delivery status."""
    from backend.app.integrations.africastalking.sms import SMSService
    sms = SMSService()
    status = await sms.check_delivery(msg_id)
    return {"message_id": status.message_id, "status": status.status}

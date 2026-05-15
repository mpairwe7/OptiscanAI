"""DHIS2 integration REST API router."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.config import settings

logger = logging.getLogger(__name__)
from fastapi import Depends
from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/dhis2",
    tags=["dhis2"],
    dependencies=[Depends(require_tier("health_system", feature="dhis2"))],
)

_client = None
_queue = None


def _is_enabled() -> bool:
    dhis2 = getattr(settings, "dhis2", None)
    if dhis2 and dhis2.enabled:
        return True
    return os.getenv("DHIS2__ENABLED", "false").lower() in ("1", "true", "yes")


def _require():
    if not _is_enabled():
        raise HTTPException(404, "DHIS2 integration not enabled")


class PatientSearchResponse(BaseModel):
    results: list[dict] = []
    count: int = 0


class ReferralRequest(BaseModel):
    org_unit: str
    tei_id: str
    event_date: str
    screening_result: str = ""
    referral_priority: str = ""
    disease_codes: str = ""


class ReferralResponse(BaseModel):
    event_id: str = ""
    status: str = "created"
    queued_offline: bool = False


@router.get("/patient/search", response_model=PatientSearchResponse)
async def search_patient(
    query: str = Query(..., min_length=1),
    org_unit: str = Query(...),
):
    _require()
    try:
        from backend.app.integrations.dhis2.client import DHIS2Client
        from backend.app.integrations.dhis2.auth import DHIS2Auth

        auth = DHIS2Auth(
            method=settings.dhis2.auth_method,
            personal_access_token=settings.dhis2.personal_access_token,
            base_url=settings.dhis2.base_url,
        )
        client = DHIS2Client(settings.dhis2.base_url, auth)
        results = await client.search_patient(query, org_unit)
        await client.close()
        return {"results": [{"tei_id": r.tei_id, "name": r.name, "org_unit": r.org_unit} for r in results], "count": len(results)}
    except Exception as e:
        raise HTTPException(502, f"DHIS2 error: {e}")


@router.post("/referral", response_model=ReferralResponse)
async def create_referral(body: ReferralRequest):
    _require()
    from backend.app.integrations.dhis2.models import ReferralEvent
    from backend.app.integrations.dhis2.offline_queue import DHIS2OfflineQueue, DHIS2Operation
    from uuid import uuid4

    referral = ReferralEvent(
        org_unit=body.org_unit,
        tei_id=body.tei_id,
        event_date=body.event_date,
        data_values={
            "screening_result": body.screening_result,
            "referral_priority": body.referral_priority,
            "disease_codes": body.disease_codes,
        },
    )

    try:
        from backend.app.integrations.dhis2.client import DHIS2Client
        from backend.app.integrations.dhis2.auth import DHIS2Auth

        auth = DHIS2Auth(method=settings.dhis2.auth_method, personal_access_token=settings.dhis2.personal_access_token, base_url=settings.dhis2.base_url)
        client = DHIS2Client(settings.dhis2.base_url, auth)
        event_id = await client.create_referral_event(referral)
        await client.close()
        return {"event_id": event_id, "status": "created"}
    except Exception as e:
        logger.warning("DHIS2 unavailable, queuing offline: %s", e)
        queue = DHIS2OfflineQueue(settings.dhis2.queue_dir)
        op = DHIS2Operation(operation_id=str(uuid4()), operation_type="create_referral", payload=referral.model_dump())
        await queue.enqueue(op)
        return {"event_id": "", "status": "queued", "queued_offline": True}


@router.get("/queue/status")
async def queue_status():
    _require()
    from backend.app.integrations.dhis2.offline_queue import DHIS2OfflineQueue
    queue = DHIS2OfflineQueue(settings.dhis2.queue_dir)
    count = await queue.get_pending_count()
    return {"pending": count}

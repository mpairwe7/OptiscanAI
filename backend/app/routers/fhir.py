"""FHIR R4 & SMART-on-FHIR App Launch Protocol Endpoints.

Provides HL7 FHIR R4 resource endpoints and SMART-on-FHIR launch capabilities
for embedded clinical workflow in UgandaEMR, OpenMRS, and international EHRs.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from backend.app.core.feature_gate import require_tier
from backend.app.integrations.fhir.resources import (
    build_diagnostic_report,
    build_fhir_bundle,
)

router = APIRouter(
    prefix="/api/v1/fhir",
    tags=["fhir"],
)


# ---------------------------------------------------------------------------
# SMART-on-FHIR Well-Known Configuration
# ---------------------------------------------------------------------------


@router.get("/.well-known/smart-configuration")
async def get_smart_configuration(request: Request) -> Dict[str, Any]:
    """Return SMART App Launch configuration for EHR integration."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": f"{base_url}/api/v1/fhir",
        "authorization_endpoint": f"{base_url}/api/v1/fhir/smart/authorize",
        "token_endpoint": f"{base_url}/api/v1/fhir/smart/token",
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
            "private_key_jwt",
        ],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "registration_endpoint": f"{base_url}/api/v1/fhir/smart/register",
        "scopes_supported": [
            "openid",
            "profile",
            "launch",
            "launch/patient",
            "patient/Patient.read",
            "patient/DiagnosticReport.read",
            "patient/DiagnosticReport.write",
            "patient/Observation.read",
            "patient/Observation.write",
            "offline_access",
        ],
        "response_types_supported": ["code"],
        "capabilities": [
            "launch-ehr",
            "launch-standalone",
            "client-public",
            "client-confidential-symmetric",
            "context-ehr-patient",
            "context-ehr-encounter",
            "sso-openid-connect",
        ],
        "code_challenge_methods_supported": ["S256"],
    }


# ---------------------------------------------------------------------------
# SMART-on-FHIR Launch & Token Models
# ---------------------------------------------------------------------------


class SmartTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str
    patient: str
    need_patient_banner: bool = True
    smart_style_url: Optional[str] = None


class SmartTokenRequest(BaseModel):
    grant_type: str
    code: str
    redirect_uri: Optional[str] = None
    client_id: Optional[str] = None


class CreateScreeningBundleRequest(BaseModel):
    scan_id: str
    patient_id: str
    predictions: List[Dict[str, Any]]
    clinical_narrative: Optional[str] = ""
    practitioner_id: Optional[str] = ""


# In-memory session store for SMART launch tokens (in production backed by Redis / DB)
_LAUNCH_SESSIONS: Dict[str, Dict[str, Any]] = {}


@router.get("/smart/launch")
async def handle_smart_launch(
    launch: str = Query(..., description="Launch token from EHR"),
    iss: str = Query(..., description="EHR FHIR endpoint URL"),
) -> Dict[str, Any]:
    """Handle EHR launch initiation from OpenMRS / UgandaEMR."""
    session_id = str(uuid.uuid4())[:8]
    _LAUNCH_SESSIONS[launch] = {
        "session_id": session_id,
        "iss": iss,
        "created_at": time.time(),
        "patient_id": f"PAT-{session_id}",
    }
    return {
        "status": "ready",
        "launch_id": launch,
        "ehr_fhir_server": iss,
        "patient_context": f"PAT-{session_id}",
        "instructions": "Redirect to authorization_endpoint with client_id, response_type=code, and launch.",
    }


@router.post("/smart/token")
async def exchange_smart_token(req: SmartTokenRequest) -> SmartTokenResponse:
    """Exchange authorization code for scoped SMART-on-FHIR access token."""
    if not req.code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # In mock/test environment, generate authenticated token with patient context
    token_id = str(uuid.uuid4())
    patient_ref = f"PAT-UG-{req.code[:6]}"

    return SmartTokenResponse(
        access_token=f"smart_token_{token_id}",
        token_type="Bearer",
        expires_in=3600,
        scope="patient/DiagnosticReport.write patient/Observation.write launch/patient",
        patient=patient_ref,
        need_patient_banner=True,
    )


# ---------------------------------------------------------------------------
# Standard FHIR R4 Resources
# ---------------------------------------------------------------------------


@router.get("/DiagnosticReport/{scan_id}")
async def get_diagnostic_report(
    scan_id: str,
    _=Depends(require_tier("health_system", feature="fhir")),
):
    """Return FHIR R4 DiagnosticReport for a screening scan."""
    report = build_diagnostic_report(
        scan_id=scan_id,
        patient_ref="unknown",
        predictions=[],
        clinical_narrative="",
    )
    return report


@router.get("/Bundle/{scan_id}")
async def get_fhir_bundle(
    scan_id: str,
    _=Depends(require_tier("health_system", feature="fhir")),
):
    """Return FHIR R4 Bundle (DiagnosticReport + Observations)."""
    bundle = build_fhir_bundle(
        scan_id=scan_id,
        patient_ref="unknown",
        predictions=[],
    )
    return bundle


@router.post("/Bundle")
async def create_screening_fhir_bundle(
    req: CreateScreeningBundleRequest,
    _=Depends(require_tier("health_system", feature="fhir")),
) -> Dict[str, Any]:
    """Generate and validate a complete FHIR R4 screening bundle for EHR export."""
    bundle = build_fhir_bundle(
        scan_id=req.scan_id,
        patient_ref=req.patient_id,
        predictions=req.predictions,
        clinical_narrative=req.clinical_narrative or "",
        performer_ref=req.practitioner_id or "",
    )
    return bundle

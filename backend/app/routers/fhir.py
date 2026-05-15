"""FHIR R4 resource endpoints for interoperability."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fastapi import Depends
from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/fhir",
    tags=["fhir"],
    dependencies=[Depends(require_tier("health_system", feature="fhir"))],
)


@router.get("/DiagnosticReport/{scan_id}")
async def get_diagnostic_report(scan_id: str):
    """Return FHIR R4 DiagnosticReport for a screening scan."""
    from backend.app.integrations.fhir.resources import build_diagnostic_report
    report = build_diagnostic_report(
        scan_id=scan_id, patient_ref="unknown", predictions=[], clinical_narrative="",
    )
    return report


@router.get("/Bundle/{scan_id}")
async def get_fhir_bundle(scan_id: str):
    """Return FHIR R4 Bundle (DiagnosticReport + Observations)."""
    from backend.app.integrations.fhir.resources import build_fhir_bundle
    bundle = build_fhir_bundle(
        scan_id=scan_id, patient_ref="unknown", predictions=[],
    )
    return bundle

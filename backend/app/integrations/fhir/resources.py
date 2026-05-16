"""FHIR R4 resource builders for screening results.

Generates DiagnosticReport and Observation resources compliant with
HL7 FHIR R4 for interoperability with Uganda eHMIS and international systems.
"""

from __future__ import annotations

import datetime

# SNOMED CT codes for common retinal diseases
SNOMED_MAP: dict[str, str] = {
    "DR": "4855003",       # Diabetic retinopathy
    "ARMD": "267718000",   # Age-related macular degeneration
    "MH": "232006008",     # Macular hole
    "BRVO": "232035008",   # Branch retinal vein occlusion
    "CRVO": "46635009",    # Central retinal vein occlusion
    "ODC": "23986001",     # Glaucoma (optic disc cupping)
    "RP": "28835009",      # Retinitis pigmentosa
    "CWS": "95731004",     # Cotton wool spots
    "ERM": "367649002",    # Epiretinal membrane
    "CSR": "312956001",    # Central serous retinopathy
}


def build_diagnostic_report(
    scan_id: str,
    patient_ref: str,
    predictions: list[dict],
    clinical_narrative: str = "",
    performer_ref: str = "",
) -> dict:
    """Build a FHIR R4 DiagnosticReport from screening result."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    report = {
        "resourceType": "DiagnosticReport",
        "id": scan_id,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "OPH",
                "display": "Ophthalmology",
            }],
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "71482-4",
                "display": "Fundus photo study",
            }],
        },
        "subject": {"reference": f"Patient/{patient_ref}"},
        "effectiveDateTime": now,
        "issued": now,
        "conclusion": clinical_narrative or "Automated retinal screening analysis",
        "result": [
            {"reference": f"Observation/{scan_id}-obs-{i}"}
            for i in range(len(predictions))
        ],
    }

    if performer_ref:
        report["performer"] = [{"reference": f"Practitioner/{performer_ref}"}]

    return report


def build_observation(
    scan_id: str,
    index: int,
    disease_code: str,
    disease_name: str,
    probability: float,
) -> dict:
    """Build a FHIR R4 Observation for a single disease finding."""
    snomed = SNOMED_MAP.get(disease_code, "")

    obs = {
        "resourceType": "Observation",
        "id": f"{scan_id}-obs-{index}",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed,
                "display": disease_name,
            }] if snomed else [],
            "text": disease_name,
        },
        "valueQuantity": {
            "value": round(probability, 4),
            "unit": "probability",
            "system": "http://unitsofmeasure.org",
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "A" if probability >= 0.5 else "N",
                "display": "Abnormal" if probability >= 0.5 else "Normal",
            }],
        }],
    }

    return obs


def build_fhir_bundle(
    scan_id: str,
    patient_ref: str,
    predictions: list[dict],
    clinical_narrative: str = "",
) -> dict:
    """Build a complete FHIR Bundle containing report + observations."""
    report = build_diagnostic_report(scan_id, patient_ref, predictions, clinical_narrative)

    entries = [{"resource": report, "fullUrl": f"DiagnosticReport/{scan_id}"}]

    for i, pred in enumerate(predictions):
        obs = build_observation(
            scan_id, i,
            pred.get("code", ""),
            pred.get("name", ""),
            pred.get("probability", 0.0),
        )
        entries.append({"resource": obs, "fullUrl": f"Observation/{scan_id}-obs-{i}"})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }

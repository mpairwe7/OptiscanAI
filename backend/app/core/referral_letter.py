"""Bilingual referral letter generator (Luganda + English).

Generates a structured referral letter from screening results with
disease findings, treatment recommendations, and urgency level.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReferralLetterData:
    """Data for generating a referral letter."""
    scan_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    patient_id: str = ""
    detected_diseases: list[str] = None
    probabilities: dict[str, float] = None
    referral_priority: str = "ROUTINE"
    risk_score: float = 0.0
    chw_name: str = ""
    facility_name: str = ""
    referral_facility: str = ""
    screening_date: str = ""
    language: str = "both"  # en | lg | both

    def __post_init__(self):
        if self.detected_diseases is None:
            self.detected_diseases = []
        if self.probabilities is None:
            self.probabilities = {}
        if not self.screening_date:
            self.screening_date = datetime.date.today().isoformat()


class ReferralLetterGenerator:
    """Generate bilingual referral letters from screening results."""

    def generate_text(self, data: ReferralLetterData) -> str:
        """Generate a plain-text referral letter."""
        sections = []

        if data.language in ("en", "both"):
            sections.append(self._english_letter(data))
        if data.language in ("lg", "both"):
            if sections:
                sections.append("\n" + "=" * 60 + "\n")
            sections.append(self._luganda_letter(data))

        return "\n".join(sections)

    def _english_letter(self, data: ReferralLetterData) -> str:
        """Generate English referral letter."""
        from backend.app.core.luganda.clinical_terms import get_disease_name, get_referral_text

        diseases_text = ""
        for code in data.detected_diseases:
            name = get_disease_name(code, "en")
            prob = data.probabilities.get(code, 0)
            diseases_text += f"  - {name} ({code}): {prob*100:.1f}% confidence\n"

        referral_text = get_referral_text(data.referral_priority, "en")

        return f"""RETINALAI CLINICAL SCREENING — REFERRAL LETTER
{'=' * 50}

Date: {data.screening_date}
Scan ID: {data.scan_id}

PATIENT INFORMATION
  Name: {data.patient_name or 'Not provided'}
  Age: {data.patient_age or 'N/A'}    Sex: {data.patient_sex or 'N/A'}
  ID: {data.patient_id or 'N/A'}

SCREENING FINDINGS
{diseases_text if diseases_text else '  No pathology detected.'}
  Overall Risk Score: {data.risk_score:.2f}

REFERRAL RECOMMENDATION
  Priority: {data.referral_priority}
  {referral_text}
  {f'Refer to: {data.referral_facility}' if data.referral_facility else ''}

SCREENED BY
  CHW: {data.chw_name or 'N/A'}
  Facility: {data.facility_name or 'N/A'}

Note: This is an AI-assisted screening result and should be confirmed
by a qualified ophthalmologist. RetinalAI v1.0.0
"""

    def _luganda_letter(self, data: ReferralLetterData) -> str:
        """Generate Luganda referral letter."""
        from backend.app.core.luganda.clinical_terms import get_disease_name, get_referral_text

        diseases_text = ""
        for code in data.detected_diseases:
            name = get_disease_name(code, "lg")
            prob = data.probabilities.get(code, 0)
            diseases_text += f"  - {name} ({code}): {prob*100:.1f}%\n"

        referral_text = get_referral_text(data.referral_priority, "lg")

        return f"""RETINALAI — EBBALUWA EY'OKUTWALA OMULWADDE
{'=' * 50}

Ennaku: {data.screening_date}
Enamba y'Okukebera: {data.scan_id}

EBIKWATA KU MULWADDE
  Erinnya: {data.patient_name or 'Tekyaweereddwa'}
  Emyaka: {data.patient_age or 'N/A'}    Ekika: {data.patient_sex or 'N/A'}

EBIVAAMU BY'OKUKEBERA
{diseases_text if diseases_text else '  Tewali bulwadde bw amaaso bulabiddwa.'}

OKUTWALA OMULWADDE
  Obunyonyi: {data.referral_priority}
  {referral_text}
  {f'Twala mu: {data.referral_facility}' if data.referral_facility else ''}

EYAKUKEBERERA
  CHW: {data.chw_name or 'N/A'}
  Eddwaliro: {data.facility_name or 'N/A'}

Obubaka: Bino bivaamu by'okukebera kw'amaaso n'ebyuma. Bisaanira
okukakasibwa omusawo w'amaaso. RetinalAI v1.0.0
"""

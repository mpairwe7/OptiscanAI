"""Multimodal confidence estimator for retinal screening.

Combines image prediction confidence, voice-extracted clinical history,
and patient demographics to produce a composite confidence score.
Graceful degradation to image-only when other inputs are missing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Modality weights
IMAGE_WEIGHT = 0.60
HISTORY_WEIGHT = 0.25
DEMOGRAPHICS_WEIGHT = 0.15

# Risk factor -> disease code boost mapping
RISK_FACTOR_BOOSTS: dict[str, dict[str, float]] = {
    "diabetes": {"DR": 0.15, "MH": 0.05, "EDN": 0.05},
    "hypertension": {"BRVO": 0.12, "CRVO": 0.10, "ODE": 0.05},
    "hiv": {"CWS": 0.15, "RT": 0.10},
    "sickle_cell": {"BRVO": 0.10},
}

# Age-related risk adjustments
AGE_RISK: dict[str, dict[str, float]] = {
    "senior": {"ARMD": 0.15, "ODC": 0.10, "DR": 0.05},  # 60+
    "middle": {"DR": 0.08, "BRVO": 0.05},                # 40-59
    "young": {"MYA": 0.10},                               # < 40
}


class MultimodalConfidenceEstimator:
    """Combine multiple modalities for enhanced screening confidence.

    When all modalities are available:
        confidence = 0.60 * image + 0.25 * history + 0.15 * demographics

    Falls back gracefully when modalities are missing.
    """

    def estimate(
        self,
        image_probabilities: dict[str, float],
        clinical_history: Optional[dict] = None,
        demographics: Optional[dict] = None,
    ) -> dict[str, float]:
        """Produce adjusted per-disease probabilities.

        Parameters
        ----------
        image_probabilities : dict
            Disease code -> probability from model inference.
        clinical_history : dict, optional
            Extracted from voice: risk_factors, symptoms, conditions.
        demographics : dict, optional
            Patient age, sex, comorbidities.

        Returns
        -------
        dict[str, float]
            Adjusted probabilities incorporating all available modalities.
        """
        adjusted = dict(image_probabilities)
        available_weight = IMAGE_WEIGHT

        # History boost
        if clinical_history and clinical_history.get("risk_factors"):
            history_boosts = self._compute_history_boosts(
                clinical_history["risk_factors"]
            )
            available_weight += HISTORY_WEIGHT
            for code, boost in history_boosts.items():
                if code in adjusted:
                    adjusted[code] = min(
                        adjusted[code] + boost * HISTORY_WEIGHT, 1.0
                    )

        # Demographics boost
        if demographics and demographics.get("age"):
            demo_boosts = self._compute_demographic_boosts(demographics)
            available_weight += DEMOGRAPHICS_WEIGHT
            for code, boost in demo_boosts.items():
                if code in adjusted:
                    adjusted[code] = min(
                        adjusted[code] + boost * DEMOGRAPHICS_WEIGHT, 1.0
                    )

        return adjusted

    def compute_multimodal_confidence(
        self,
        image_confidence: float,
        history_available: bool = False,
        demographics_available: bool = False,
    ) -> float:
        """Compute overall multimodal confidence score."""
        if history_available and demographics_available:
            return image_confidence * IMAGE_WEIGHT + 0.8 * HISTORY_WEIGHT + 0.7 * DEMOGRAPHICS_WEIGHT
        if history_available:
            return image_confidence * (IMAGE_WEIGHT + DEMOGRAPHICS_WEIGHT) + 0.8 * HISTORY_WEIGHT
        return image_confidence

    def _compute_history_boosts(self, risk_factors: dict) -> dict[str, float]:
        """Compute per-disease boosts from clinical risk factors."""
        boosts: dict[str, float] = {}
        for factor, present in risk_factors.items():
            if present and factor in RISK_FACTOR_BOOSTS:
                for code, boost in RISK_FACTOR_BOOSTS[factor].items():
                    boosts[code] = boosts.get(code, 0) + boost
        return boosts

    def _compute_demographic_boosts(self, demographics: dict) -> dict[str, float]:
        """Compute per-disease boosts from patient demographics."""
        age = demographics.get("age", 0)
        try:
            age = int(age)
        except (ValueError, TypeError):
            return {}

        if age >= 60:
            group = "senior"
        elif age >= 40:
            group = "middle"
        else:
            group = "young"

        return AGE_RISK.get(group, {})

    def get_urgency_score(
        self,
        detected_diseases: list[str],
        risk_factors: Optional[dict] = None,
    ) -> int:
        """Map to MoH urgency scale (1-5).

        5 = EMERGENCY (sight-threatening, immediate referral)
        4 = URGENT (refer within 24 hours)
        3 = ROUTINE (refer within 1 week)
        2 = FOLLOW_UP (schedule routine follow-up)
        1 = NORMAL (no pathology detected)
        """
        if not detected_diseases:
            return 1

        emergency = {"CRAO", "CRVO", "AION", "ODE"}
        urgent = {"DR", "BRVO", "RT", "ODC"}

        disease_set = set(detected_diseases)

        if disease_set & emergency:
            return 5
        if disease_set & urgent:
            return 4

        # Risk factor escalation
        if risk_factors:
            if risk_factors.get("diabetes") and "DR" in disease_set:
                return 4
            if risk_factors.get("hypertension") and disease_set & {"BRVO", "CRVO"}:
                return 4

        if len(detected_diseases) >= 3:
            return 3

        return 2 if detected_diseases else 1

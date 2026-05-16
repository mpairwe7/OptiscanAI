"""Voice history extractor for multimodal screening.

Parses voice transcripts to extract structured clinical data:
symptoms, duration, known conditions, medications. Uses Claude
for structured extraction with regex-based keyword fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Uganda-specific medical term patterns
CONDITION_PATTERNS: dict[str, list[str]] = {
    "diabetes": [
        "diabetes",
        "diabetic",
        "sugar disease",
        "sugar",
        "esukaali",
        "sukaali",
        "insulin",
    ],
    "hypertension": ["hypertension", "high blood pressure", "high bp", "pressure", "puleesa"],
    "hiv": ["hiv", "aids", "silimu", "arv", "antiretroviral"],
    "sickle_cell": ["sickle cell", "sickler", "sickle"],
    "malaria": ["malaria", "omusujja", "ensiri"],
    "tuberculosis": ["tuberculosis", "tb", "cough"],
}

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "blurry_vision": [
        "blurry",
        "blur",
        "can't see",
        "cannot see",
        "foggy",
        "ebyenzirikizi",
        "okulaba",
    ],
    "eye_pain": ["eye pain", "eyes hurt", "painful eyes", "okubabuka"],
    "headache": ["headache", "head ache", "head pain", "omutwe"],
    "vision_loss": ["blind", "losing sight", "can't see anything", "darkness"],
    "floaters": ["floaters", "spots", "flies", "dots"],
    "flashing_lights": ["flashing", "lightning", "sparks"],
    "eye_redness": ["red eye", "red eyes", "bloodshot"],
    "double_vision": ["double vision", "seeing double"],
}

DURATION_PATTERNS = [
    (r"(\d+)\s*(?:day|days|ennaku)", "days"),
    (r"(\d+)\s*(?:week|weeks|wiiki|sabiiti)", "weeks"),
    (r"(\d+)\s*(?:month|months|emyezi)", "months"),
    (r"(\d+)\s*(?:year|years|emyaka)", "years"),
    (r"long\s*time|for\s*long|emyaka\s*mingi", "chronic"),
]


class VoiceHistoryExtractor:
    """Extract structured clinical data from voice transcripts."""

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm = llm_client

    async def extract(self, transcript: str) -> dict:
        """Extract clinical data — tries LLM first, falls back to rules."""
        if self._llm:
            try:
                return await self.extract_with_llm(transcript)
            except Exception as e:
                logger.warning("LLM extraction failed, falling back to rules: %s", e)

        return self.extract_with_rules(transcript)

    async def extract_with_llm(self, transcript: str) -> dict:
        """Claude-based structured extraction."""
        prompt = f"""Extract structured clinical information from this patient transcript.
Return JSON with these fields:
- symptoms: list of symptom descriptions
- conditions: list of known medical conditions
- medications: list of current medications
- duration: how long symptoms have been present
- severity: mild/moderate/severe
- risk_factors: diabetes, hypertension, hiv, sickle_cell, etc.

Transcript: "{transcript}"

Return ONLY valid JSON, no other text."""

        # Use the project's LLM interface
        from src.agents.llm import call_llm

        response = await call_llm(prompt, max_tokens=500)

        import json

        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return self.extract_with_rules(transcript)

    def extract_with_rules(self, transcript: str) -> dict:
        """Keyword/regex-based extraction for offline operation."""
        lower = transcript.lower()

        # Extract conditions
        conditions = []
        risk_factors = {}
        for condition, patterns in CONDITION_PATTERNS.items():
            if any(p in lower for p in patterns):
                conditions.append(condition)
                risk_factors[condition] = True

        # Extract symptoms
        symptoms = []
        for symptom, patterns in SYMPTOM_PATTERNS.items():
            if any(p in lower for p in patterns):
                symptoms.append(symptom)

        # Extract duration
        duration = None
        for pattern, unit in DURATION_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                if unit == "chronic":
                    duration = "chronic"
                else:
                    val = match.group(1) if match.lastindex else ""
                    duration = f"{val} {unit}"
                break

        # Estimate severity from keyword intensity
        severity = "mild"
        severe_words = ["very", "severe", "terrible", "worst", "nnyo", "nnyangu"]
        moderate_words = ["some", "little", "moderate", "bit"]
        if any(w in lower for w in severe_words):
            severity = "severe"
        elif any(w in lower for w in moderate_words):
            severity = "moderate"

        return {
            "symptoms": symptoms,
            "conditions": conditions,
            "medications": [],
            "duration": duration,
            "severity": severity,
            "risk_factors": risk_factors,
            "extraction_method": "rules",
        }

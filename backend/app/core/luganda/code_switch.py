"""Code-switching handler for mixed Luganda/English ASR output.

Detects language segments and maps medical terms from either language
to canonical disease codes. Handles common Ugandan English patterns
(e.g., "sugar disease" -> diabetes -> DR).
"""

from __future__ import annotations

# Ugandan English colloquial -> canonical medical term
UGANDAN_ENGLISH_MAP: dict[str, str] = {
    "sugar disease": "diabetes",
    "sugar": "diabetes",
    "high sugar": "diabetes",
    "pressure": "hypertension",
    "high pressure": "hypertension",
    "bp": "hypertension",
    "slim": "hiv",
    "the virus": "hiv",
    "sickle": "sickle_cell",
    "sickler": "sickle_cell",
    "malaria": "malaria",
    "headache": "headache",
    "blurry": "blurry_vision",
    "can't see well": "blurry_vision",
    "eye pain": "eye_pain",
    "eyes paining": "eye_pain",
    "red eyes": "eye_redness",
}

# Luganda terms -> canonical medical term
LUGANDA_TERM_MAP: dict[str, str] = {
    "esukaali": "diabetes",
    "sukaali": "diabetes",
    "puleesa": "hypertension",
    "pulesa": "hypertension",
    "silimu": "hiv",
    "omusujja": "malaria",
    "ensiri": "malaria",
    "omusaayi": "blood_condition",
    "amaaso": "eye_condition",
    "okubabuka": "eye_pain",
    "omutwe": "headache",
    "okulaba": "vision_issue",
    "ebyenzirikizi": "blurry_vision",
}

# Canonical term -> related disease codes
TERM_TO_DISEASE: dict[str, list[str]] = {
    "diabetes": ["DR"],
    "hypertension": ["BRVO", "CRVO"],
    "hiv": ["CWS", "RT"],
    "sickle_cell": ["BRVO"],
    "eye_pain": [],
    "blurry_vision": [],
    "headache": [],
    "malaria": [],
}


def detect_language_segments(text: str) -> list[dict]:
    """Detect language segments in mixed Luganda/English text.

    Returns list of segments with language annotation.
    Simple heuristic: Luganda words have specific patterns
    (e.g., prefixes oku-, obu-, emu-, aba-, ama-).
    """
    words = text.split()
    segments = []
    current_lang = "en"
    current_words = []

    luganda_prefixes = (
        "oku", "obu", "emu", "aba", "ama", "eby", "eki", "omu",
        "enk", "ens", "ebb", "ekk", "enn",
    )

    for word in words:
        lower = word.lower().strip(".,!?")
        is_luganda = (
            any(lower.startswith(p) for p in luganda_prefixes)
            or lower in LUGANDA_TERM_MAP
        )

        detected = "lg" if is_luganda else "en"

        if detected != current_lang and current_words:
            segments.append({
                "language": current_lang,
                "text": " ".join(current_words),
            })
            current_words = []
            current_lang = detected

        current_words.append(word)

    if current_words:
        segments.append({
            "language": current_lang,
            "text": " ".join(current_words),
        })

    return segments


def extract_medical_terms(text: str) -> list[dict]:
    """Extract canonical medical terms from mixed-language text.

    Returns list of detected terms with source language.
    """
    lower = text.lower()
    found = []

    # Check Ugandan English patterns
    for pattern, canonical in UGANDAN_ENGLISH_MAP.items():
        if pattern in lower:
            found.append({
                "term": canonical,
                "source": pattern,
                "language": "en",
                "disease_codes": TERM_TO_DISEASE.get(canonical, []),
            })

    # Check Luganda terms
    for pattern, canonical in LUGANDA_TERM_MAP.items():
        if pattern in lower:
            found.append({
                "term": canonical,
                "source": pattern,
                "language": "lg",
                "disease_codes": TERM_TO_DISEASE.get(canonical, []),
            })

    # Deduplicate by canonical term
    seen = set()
    unique = []
    for item in found:
        if item["term"] not in seen:
            seen.add(item["term"])
            unique.append(item)

    return unique


def extract_risk_factors(text: str) -> dict[str, bool]:
    """Extract clinical risk factors from transcript text."""
    lower = text.lower()
    factors = {}

    diabetes_terms = ["sugar", "diabetes", "esukaali", "sukaali", "insulin"]
    if any(t in lower for t in diabetes_terms):
        factors["diabetes"] = True

    htn_terms = ["pressure", "hypertension", "puleesa", "pulesa", "bp"]
    if any(t in lower for t in htn_terms):
        factors["hypertension"] = True

    hiv_terms = ["hiv", "aids", "silimu", "arv", "slim"]
    if any(t in lower for t in hiv_terms):
        factors["hiv"] = True

    sickle_terms = ["sickle", "sickler"]
    if any(t in lower for t in sickle_terms):
        factors["sickle_cell"] = True

    return factors

"""Luganda phoneme mapping for Piper TTS.

Provides pronunciation rules for clinical terms, number reading,
and date formatting in Luganda for natural TTS output.
"""

from __future__ import annotations

# Luganda vowel system
VOWELS = {"a", "e", "i", "o", "u"}

# Luganda consonant clusters common in medical terms
CONSONANT_CLUSTERS = {
    "mb", "nd", "ng", "nk", "nt", "mp", "nz", "ny", "bw", "gw", "kw", "lw",
    "mw", "nw", "pw", "sw", "tw", "zw",
}

# Pronunciation overrides for medical terms
MEDICAL_PRONUNCIATION: dict[str, str] = {
    "DR": "dee-aa",
    "ARMD": "eh-aa-em-dee",
    "BRVO": "bee-aa-vee-oh",
    "CRVO": "see-aa-vee-oh",
    "HIV": "echi-ai-vee",
    "AIDS": "edzi",
    "MoH": "em-oh-echi",
    "DHIS2": "dee-echi-ai-es-bbiri",
    "RetinalAI": "retinala-ai",
}

# Number reading in Luganda
LUGANDA_NUMBERS: dict[int, str] = {
    0: "zeero",
    1: "emu",
    2: "bbiri",
    3: "ssatu",
    4: "nnya",
    5: "ttaano",
    6: "mukaaga",
    7: "musanvu",
    8: "munaana",
    9: "mwenda",
    10: "kkumi",
    20: "amakumi abiri",
    30: "amakumi asatu",
    40: "amakumi ana",
    50: "amakumi ataano",
    60: "amakumi mukaaga",
    70: "amakumi musanvu",
    80: "amakumi munaana",
    90: "amakumi mwenda",
    100: "kikumi",
}


def number_to_luganda(n: int) -> str:
    """Convert a number to Luganda words."""
    if n in LUGANDA_NUMBERS:
        return LUGANDA_NUMBERS[n]
    if n < 0:
        return f"obubi {number_to_luganda(-n)}"
    if n < 10:
        return LUGANDA_NUMBERS.get(n, str(n))
    if n < 20:
        return f"kkumi n'{LUGANDA_NUMBERS.get(n - 10, str(n - 10))}"
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return LUGANDA_NUMBERS.get(tens, str(tens))
        return f"{LUGANDA_NUMBERS.get(tens, str(tens))} mu {LUGANDA_NUMBERS.get(ones, str(ones))}"
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        h_word = f"bikumi {LUGANDA_NUMBERS.get(hundreds, str(hundreds))}" if hundreds > 1 else "kikumi"
        if remainder == 0:
            return h_word
        return f"{h_word} mu {number_to_luganda(remainder)}"
    return str(n)


def percentage_to_luganda(value: float) -> str:
    """Convert a percentage to Luganda speech."""
    rounded = round(value)
    return f"pasenti {number_to_luganda(rounded)}"


def format_tts_text(text: str, language: str = "lg") -> str:
    """Format text for natural TTS output.

    Expands abbreviations, numbers, and medical codes for speech.
    """
    if language != "lg":
        return text

    # Expand medical abbreviations
    for abbr, pronunciation in MEDICAL_PRONUNCIATION.items():
        text = text.replace(abbr, pronunciation)

    # Expand percentages
    import re
    def replace_pct(match):
        num = float(match.group(1))
        return percentage_to_luganda(num)

    text = re.sub(r"(\d+(?:\.\d+)?)%", replace_pct, text)

    return text

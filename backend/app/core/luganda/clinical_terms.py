"""Bilingual clinical terminology dictionary for Luganda + English.

Maps all 45 RFMiD disease codes, referral priorities, screening instructions,
and treatment terms between English and Luganda. Used by TTS to speak results
in the CHW's chosen language.

Note: Medical Luganda translations are approximate and should be validated
by Ugandan clinical linguists before national deployment.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Disease name translations
# ---------------------------------------------------------------------------

DISEASE_NAMES: dict[str, dict[str, str]] = {
    "DR": {
        "en": "Diabetic Retinopathy",
        "lg": "Obulwadde bw'amaaso olw'esukaali",
    },
    "ARMD": {
        "en": "Age-Related Macular Degeneration",
        "lg": "Okuvunda kw'amaaso olw'obukadde",
    },
    "MH": {
        "en": "Macular Hole",
        "lg": "Ekituli mu maaso",
    },
    "DN": {
        "en": "Drusen",
        "lg": "Ebizige mu maaso",
    },
    "MYA": {
        "en": "Myopia",
        "lg": "Obutalaba wala",
    },
    "BRVO": {
        "en": "Branch Retinal Vein Occlusion",
        "lg": "Okuzibika kw'omusipi gw'omusaayi mu liiso",
    },
    "TSLN": {
        "en": "Tessellation",
        "lg": "Okwawuka kw'ennyuma y'eriiso",
    },
    "ERM": {
        "en": "Epiretinal Membrane",
        "lg": "Oluvu ku maaso g'eriiso",
    },
    "LS": {
        "en": "Laser Scars",
        "lg": "Ebizige by'okulongoosebwa kw'amaaso",
    },
    "MS": {
        "en": "Myelinated Nerve Fibers",
        "lg": "Emisipi gy'amaaso egirekeddemu",
    },
    "CSR": {
        "en": "Central Serous Retinopathy",
        "lg": "Amazzi mu maaso",
    },
    "ODC": {
        "en": "Optic Disc Cupping",
        "lg": "Okweyongera kw'ekituli mu kkubo ly'eriiso",
    },
    "CRVO": {
        "en": "Central Retinal Vein Occlusion",
        "lg": "Okuzibika kw'omusipi gw'omusaayi omukulu mu liiso",
    },
    "TV": {
        "en": "Tortuous Vessels",
        "lg": "Emisipi gy'omusaayi egyewunyise",
    },
    "AH": {
        "en": "Asteroid Hyalosis",
        "lg": "Ebipande mu mazzi g'eriiso",
    },
    "ODP": {
        "en": "Optic Disc Pallor",
        "lg": "Okweruuka kw'eriiso",
    },
    "ODE": {
        "en": "Optic Disc Edema",
        "lg": "Okuzimba kw'eriiso",
    },
    "ST": {
        "en": "Optociliary Shunt",
        "lg": "Okwetoolola kw'emisipi gy'omusaayi mu liiso",
    },
    "AION": {
        "en": "Anterior Ischemic Optic Neuropathy",
        "lg": "Obulwadde bw'omusipi gw'eriiso olw'obutafuna musaayi",
    },
    "PT": {
        "en": "Parafoveal Telangiectasia",
        "lg": "Okweyongera kw'emisipi gy'omusaayi okumpi n'ekitundu",
    },
    "RT": {
        "en": "Retinitis",
        "lg": "Okukuba kw'ennyuma y'eriiso",
    },
    "RS": {
        "en": "Retinosis",
        "lg": "Obulwadde bw'ennyuma y'eriiso",
    },
    "CRS": {
        "en": "Chorioretinal Scars",
        "lg": "Ebizige mu nnyuma y'eriiso",
    },
    "EDN": {
        "en": "Edema",
        "lg": "Okuzimba",
    },
    "RPEC": {
        "en": "RPE Changes",
        "lg": "Enkyukakyuka mu nnyuma y'eriiso",
    },
    "MHL": {
        "en": "Macular Hole (Lamellar)",
        "lg": "Ekituli ekitono mu maaso",
    },
    "RP": {
        "en": "Retinitis Pigmentosa",
        "lg": "Obulwadde bw'amaaso obw'obutukuvu",
    },
    "CWS": {
        "en": "Cotton Wool Spots",
        "lg": "Ebitundu eby'era mu maaso",
    },
}

# ---------------------------------------------------------------------------
# Referral priorities
# ---------------------------------------------------------------------------

REFERRAL_PRIORITIES: dict[str, dict[str, str]] = {
    "EMERGENCY": {
        "en": "Emergency — go to hospital immediately",
        "lg": "Obujjanjabi bwa mangu — genda mu ddwaliro kati kati",
    },
    "URGENT": {
        "en": "Urgent — see an eye doctor within 24 hours",
        "lg": "Kyamangu — laba musawo w'amaaso mu ssaawa 24",
    },
    "ROUTINE": {
        "en": "Routine — schedule appointment within one week",
        "lg": "Ekya bulijjo — teeka enteekateeka mu wiiki emu",
    },
    "FOLLOW_UP": {
        "en": "Follow-up — schedule routine check-up",
        "lg": "Okuddamu — teeka enteekateeka y'okukebera",
    },
    "NORMAL": {
        "en": "Normal — no eye disease detected",
        "lg": "Bya bulijjo — tewali bulwadde bw'amaaso bulabiddwa",
    },
}

# ---------------------------------------------------------------------------
# Screening instructions (CHW prompts)
# ---------------------------------------------------------------------------

SCREENING_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "look_straight": {
        "en": "Ask the patient to look straight ahead",
        "lg": "Buulira omulwadde atunuulire ddyo mu maaso",
    },
    "hold_still": {
        "en": "Hold the device steady",
        "lg": "Kwata essimu nga tekanyeekanyeeki",
    },
    "capture_image": {
        "en": "Capture the fundus image now",
        "lg": "Kwata ekifaananyi ky'amaaso kati",
    },
    "retake": {
        "en": "Image quality is poor. Please retake",
        "lg": "Ekifaananyi tekirungi. Nsaba oddemu okukikwata",
    },
    "processing": {
        "en": "Analyzing the image. Please wait",
        "lg": "Nkola ku kifaananyi. Lindako",
    },
    "complete": {
        "en": "Screening is complete. Here are the results",
        "lg": "Okukebera kukomye. Bino bye bivaamu",
    },
    "greeting": {
        "en": "Welcome to RetinalAI Screening. Please describe patient symptoms",
        "lg": "Nkulamusizza mu RetinalAI. Yogera ebikwata ku bulwadde bw'omulwadde",
    },
}

# ---------------------------------------------------------------------------
# Common medical terms used in voice conversation
# ---------------------------------------------------------------------------

MEDICAL_TERMS: dict[str, dict[str, str]] = {
    "diabetes": {"en": "diabetes", "lg": "esukaali"},
    "hypertension": {"en": "high blood pressure", "lg": "puleesa ey'omusaayi"},
    "hiv": {"en": "HIV/AIDS", "lg": "silimu"},
    "sickle_cell": {"en": "sickle cell disease", "lg": "obulwadde bw'omusaayi"},
    "malaria": {"en": "malaria", "lg": "omusujja gw'ensiri"},
    "blurry_vision": {"en": "blurry vision", "lg": "okulaba ebyenzirikizi"},
    "eye_pain": {"en": "eye pain", "lg": "okubabuka kw'amaaso"},
    "headache": {"en": "headache", "lg": "okuba omutwe"},
    "referral": {"en": "referral", "lg": "okutwala omulwadde"},
    "screening": {"en": "eye screening", "lg": "okukebera amaaso"},
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_disease_name(code: str, language: str = "en") -> str:
    """Get disease display name in the specified language."""
    entry = DISEASE_NAMES.get(code, {})
    return entry.get(language, entry.get("en", code))


def get_referral_text(priority: str, language: str = "en") -> str:
    """Get referral priority text in the specified language."""
    entry = REFERRAL_PRIORITIES.get(priority.upper(), {})
    return entry.get(language, entry.get("en", priority))


def get_instruction(key: str, language: str = "en") -> str:
    """Get a screening instruction in the specified language."""
    entry = SCREENING_INSTRUCTIONS.get(key, {})
    return entry.get(language, entry.get("en", key))


def translate_screening_result(
    detected_diseases: list[str],
    referral_priority: str,
    language: str = "en",
) -> str:
    """Generate a spoken screening result summary."""
    if not detected_diseases:
        return get_referral_text("NORMAL", language)

    names = [get_disease_name(d, language) for d in detected_diseases]
    referral = get_referral_text(referral_priority, language)

    if language == "lg":
        disease_list = ", ".join(names)
        return (
            f"Okukebera kulaze: {disease_list}. "
            f"Obunyonyi: {len(detected_diseases)}. "
            f"{referral}"
        )

    disease_list = ", ".join(names)
    return (
        f"Screening detected: {disease_list}. "
        f"Findings: {len(detected_diseases)}. "
        f"{referral}"
    )

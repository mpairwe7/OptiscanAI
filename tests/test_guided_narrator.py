"""Tests for Guided Clinical Narrator and Anti-Hallucination Guardrails."""


from src.narrator.guided_narrator import (
    MANDATORY_DISCLOSURE,
    ClinicalGuardrailEngine,
    FindingDetail,
    GuidedClinicalNarrator,
    ReferralPriority,
    StructuredClinicalReport,
)


def test_guardrail_sanitizes_prohibited_phrases():
    guard = ClinicalGuardrailEngine()
    alarmist_text = (
        "Patient shows diabetic lesions with life-threatening complications and "
        "emergency surgery is needed to prevent permanent vision loss."
    )
    cleaned, violations = guard.sanitize_text(alarmist_text)
    assert len(violations) >= 3
    assert "life-threatening" not in cleaned.lower()
    assert "emergency surgery" not in cleaned.lower()
    assert "permanent vision loss" not in cleaned.lower()


def test_guardrail_verifies_grounding():
    guard = ClinicalGuardrailEngine()
    narrative = "Examination suggests severe glaucoma and diabetic retinopathy."
    # Grounded case
    res1 = guard.verify_grounding(narrative, detected_codes={"GLC", "DR"}, detected_names={"Glaucoma", "Diabetic Retinopathy"})
    assert res1["is_grounded"]
    assert len(res1["hallucinated_diseases"]) == 0

    # Hallucinated case
    res2 = guard.verify_grounding(narrative, detected_codes={"MYA"}, detected_names={"Myopia"})
    assert not res2["is_grounded"]
    assert "glaucoma" in res2["hallucinated_diseases"] or "diabetic retinopathy" in res2["hallucinated_diseases"]


def test_structured_report_mandates_disclosure():
    report = StructuredClinicalReport(
        findings=[FindingDetail(code="DR", name="Diabetic Retinopathy", confidence=0.88)],
        referral_priority=ReferralPriority.URGENT_SPECIALIST,
        clinical_summary="Diabetic retinopathy confirmed with microaneurysms.",
        recommendation="Urgent specialist referral.",
        ai_disclosure="custom incomplete statement",
    )
    assert MANDATORY_DISCLOSURE == report.ai_disclosure


def test_guided_narrator_generation_and_prose():
    narrator = GuidedClinicalNarrator()
    preds = [
        {"code": "DR", "name": "Diabetic Retinopathy", "probability": 0.92},
        {"code": "ODC", "name": "Optic Disc Cupping", "probability": 0.74},
    ]
    report = narrator.generate_grounded_report(preds, referral_priority="URGENT_SPECIALIST")
    assert len(report.findings) == 2
    assert report.referral_priority == ReferralPriority.URGENT_SPECIALIST
    assert MANDATORY_DISCLOSURE in report.ai_disclosure

    prose = narrator.to_clinical_prose(report)
    assert "Diabetic Retinopathy" in prose
    assert "URGENT_SPECIALIST" in prose
    assert MANDATORY_DISCLOSURE in prose

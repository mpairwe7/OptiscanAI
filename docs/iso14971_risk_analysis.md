# ISO 14971 Risk Analysis — RetinalAI Clinical Screening Platform

## Document Control

| Field | Value |
|-------|-------|
| Product | RetinalAI Clinical Screening Platform v1.0.0 |
| Standard | ISO 14971:2019 — Application of risk management to medical devices |
| Classification | IEC 62304 Class B (non-serious injury potential with mitigation) |
| Date | 2026-05-07 |
| Status | Draft — Pending MoH Technical Review |

---

## 1. Intended Use

AI-assisted screening tool for multi-label retinal disease detection from fundus photographs. Operated by trained Community Health Workers (CHWs) in primary healthcare settings in Uganda. **Not a diagnostic device** — all positive findings require confirmation by a qualified ophthalmologist.

### 1.1 Intended Users
- Community Health Workers (CHWs) with basic training (5-day curriculum)
- Clinical officers at primary care facilities
- Ophthalmologists (for result review and confirmation)

### 1.2 Intended Environment
- Primary healthcare centres in rural Uganda
- Intermittent or no internet connectivity (offline-first design)
- Mid-range Android devices (4GB RAM minimum)
- Variable lighting conditions (clinic fluorescent, outdoor, dim indoor)

---

## 2. Hazard Identification

| ID | Hazard | Hazardous Situation | Harm |
|----|--------|---------------------|------|
| H-01 | False negative (missed disease) | Patient with sight-threatening DR not referred | Delayed treatment, potential vision loss |
| H-02 | False positive (over-referral) | Healthy patient referred unnecessarily | Unnecessary anxiety, wasted resources, loss of trust |
| H-03 | Fundus gate failure | Non-fundus image accepted for inference | Incorrect screening result |
| H-04 | Offline data loss | Prediction audit trail corrupted or lost | Regulatory non-compliance, untraceable decisions |
| H-05 | Model drift | Performance degrades on new population data | Systematic misclassification |
| H-06 | Wrong patient association | Results linked to incorrect patient ID | Treatment given to wrong person |
| H-07 | Consent violation | Patient data processed without valid consent | PDP Act 2019 violation |
| H-08 | Adversarial input | Manipulated image bypasses gate | Arbitrary classification output |
| H-09 | Device degradation | Low-quality phone camera produces poor images | Reduced gate/model accuracy |
| H-10 | Language miscommunication | Luganda clinical terms mistranslated | CHW takes wrong action |
| H-11 | Connectivity-dependent failure | System fails silently when offline | CHW believes screening unavailable |
| H-12 | Referral not completed | Referral generated but patient does not attend | Disease progression without treatment |

---

## 3. Risk Estimation & Evaluation

| ID | Severity | Probability | Risk Level | Acceptable? |
|----|----------|-------------|------------|-------------|
| H-01 | Serious | Occasional | HIGH | No — requires mitigation |
| H-02 | Minor | Probable | MEDIUM | Acceptable with monitoring |
| H-03 | Moderate | Remote | LOW | Acceptable |
| H-04 | Moderate | Remote | LOW | Acceptable |
| H-05 | Serious | Occasional | HIGH | No — requires mitigation |
| H-06 | Serious | Remote | MEDIUM | Acceptable with controls |
| H-07 | Moderate | Occasional | MEDIUM | Acceptable with controls |
| H-08 | Moderate | Improbable | LOW | Acceptable |
| H-09 | Moderate | Occasional | MEDIUM | Acceptable with controls |
| H-10 | Moderate | Remote | LOW | Acceptable |
| H-11 | Minor | Occasional | LOW | Acceptable |
| H-12 | Serious | Probable | HIGH | No — requires mitigation |

---

## 4. Risk Control Measures

### H-01: False Negative (Missed Disease) — HIGH RISK
| Control | Implementation | Verification |
|---------|---------------|-------------|
| Per-class precision-floor thresholds (min 0.10) | `src/models/retinal_foundation_hybrid_v2.py` — threshold optimization | Validated in `tests/test_hybrid_v2.py` |
| Test-Time Augmentation (6 views) | `predict_with_tta()` in teacher model | Reduces false negatives by averaging |
| Mandatory human review for uncertain cases | Human-in-the-loop review queue | `backend/app/routers/review.py` |
| "AI-assisted screening — confirm with ophthalmologist" disclaimer | Displayed on all results + referral letters | UI + `referral_letter.py` |
| Continuous performance monitoring | Drift detection (PSI + KS test) | `backend/app/core/drift_detector.py` |

### H-05: Model Drift — HIGH RISK
| Control | Implementation | Verification |
|---------|---------------|-------------|
| Automated drift detection | PSI + KS test on prediction distributions | `drift_detector.py`, runs every 1000 predictions |
| Active learning loop | Low-confidence samples flagged for expert review | `active_learning.py` |
| Federated learning updates | Privacy-preserving model improvement from clinic data | `federated_client.py` (FlowerLoRAClient) |
| Quarterly re-validation | Scheduled clinical validation on held-out set | Post-market plan in MoH submission |

### H-12: Referral Not Completed — HIGH RISK
| Control | Implementation | Verification |
|---------|---------------|-------------|
| SMS referral notification | Sent to patient + facility | `africastalking/sms.py` |
| DHIS2 referral tracking | Referral event in national system | `dhis2/client.py` |
| Mobile money transport support | Payment for transport to referral facility | `mobile_money/client.py` |
| Referral completion rate monitoring | Tracked in DHIS2 aggregate reporting | Target: >= 60% completion |

---

## 5. Residual Risk Assessment

After all controls are applied:

| ID | Residual Risk | Acceptable? |
|----|--------------|-------------|
| H-01 | LOW (precision floors + human review + disclaimer) | Yes |
| H-05 | LOW (drift detection + federated updates + quarterly review) | Yes |
| H-12 | MEDIUM (SMS + DHIS2 + mobile money, but patient compliance variable) | Yes, with monitoring |

**Overall residual risk: ACCEPTABLE** — Benefits of early retinal disease detection in underserved populations substantially outweigh residual risks when used as intended (screening tool with mandatory specialist confirmation).

---

## 6. Post-Market Surveillance Plan

1. **Continuous monitoring**: Drift detection on every 1000th prediction
2. **Quarterly clinical review**: Re-validate on held-out Ugandan fundus images
3. **Bias audit**: Monthly automated audit across device/lighting/geographic subgroups
4. **Incident reporting**: Automated alerts when F1 drops >5% from baseline
5. **User feedback**: CHW satisfaction surveys at pilot sites
6. **Regulatory updates**: Annual review against evolving Uganda NDA and MoH guidelines

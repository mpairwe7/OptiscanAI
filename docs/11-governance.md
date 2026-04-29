# Governance & Compliance

Model cards, dataset cards, fairness evaluation, automated bias auditing, immutable audit trail, active learning, and human-in-the-loop review. Designed for EU AI Act (August 2026) and FDA SaMD compliance.

> **V2 additions**: `BiasAuditor` (automated demographic parity/equalized odds), `ImmutableAuditLogger` (SHA-256 hash-chain), `ActiveLearningManager` (uncertainty-based flagging + ophthalmologist review loop), precision-floor threshold optimization.

## Regulatory Context

Retinal disease classification is classified as **high-risk AI** under:

| Regulation | Requirement | Status |
|---|---|---|
| EU AI Act (2024/1689) | Documented risk management, data governance, human oversight | Implemented |
| EU MDR 2017/745 | Software as Medical Device (SaMD) lifecycle management | Pre-conformity |
| FDA 21 CFR Part 820 | Quality system for AI/ML medical devices | Pre-conformity |
| ISO 14971 | Risk management for medical devices | Risk documentation framework in place |
| IEC 62304 | Medical device software lifecycle | Lifecycle tracking via audit trail |

## Model Cards

Standardized documentation following [Google's Model Cards framework](https://modelcards.withgoogle.com/) for every trained model.

### Contents

- **Model Details**: Architecture, parameters, training date, framework, license
- **Intended Use**: Primary use case, target users, out-of-scope uses
- **Training Data**: Dataset, samples, classes, preprocessing, known limitations
- **Performance**: F1, AUC-ROC, mAP, precision, recall, hamming loss, inference latency
- **Fairness Analysis**: Subgroup performance evaluation results
- **Ethical Considerations**: Risks and mitigations for medical AI
- **Regulatory Info**: Classification, applicable regulations, conformity status

### Generating Model Cards

```bash
# Generate both model card and dataset card
make model-card

# Custom paths
PYTHONPATH=. python3 scripts/generate_model_card.py \
  --config configs/train.yaml \
  --metrics outputs/evaluation_metrics.json \
  --output-dir outputs/governance
```

**Output**: `outputs/governance/MODEL_CARD.md` + `model_card.json`

Implementation: `src/governance/model_card.py`

## Dataset Cards

Following [Gebru et al. (2021) Datasheets for Datasets](https://arxiv.org/abs/1803.09010):

- **Motivation**: Purpose, creators, funding
- **Composition**: Samples, classes, format, class distribution, multi-label statistics
- **Collection**: Process, time period, geography, consent
- **Preprocessing**: Resize, normalization, split ratios
- **Uses**: Intended use, not suitable for
- **Known Issues**: Class imbalance, demographic gaps, quality variation

Auto-populated from training data via `populate_from_dataframe()`.

**Output**: `outputs/governance/DATASET_CARD.md` + `dataset_card.json`

Implementation: `src/governance/dataset_card.py`

## Fairness Evaluation

Evaluates model performance parity across disease categories and prevalence groups.

### Disease Category Fairness

Groups diseases into clinical categories and measures performance disparity:

| Category | Diseases |
|---|---|
| VASCULAR | DR, BRVO, CRVO, CRAO, BRAO, HR, PRH, VH, MCA, VS |
| DEGENERATIVE | ARMD, MH, DN, MYA, ERM, MHL, RP |
| GLAUCOMATOUS | ODC, ODP, ODE, ODPM |
| INFLAMMATORY | RS, CRS, CWS, CB, RPEC |
| OTHER | Remaining 19 diseases |

### Prevalence Fairness

Splits diseases by sample count:

| Bucket | Criteria |
|---|---|
| COMMON | > 5% of dataset |
| MODERATE | 1-5% of dataset |
| RARE | < 1% of dataset |

### Metrics

- **Max F1 Disparity**: Difference between best and worst subgroup F1
- **Max AUC Disparity**: Difference between best and worst subgroup AUC-ROC
- **Equalized Odds**: Satisfied if F1 disparity < 0.10
- **Automated Recommendations**: Flagged when disparity exceeds thresholds

Implementation: `src/governance/fairness.py`

## Audit Trail

Immutable, append-only log of all model lifecycle events with chained SHA-256 checksums for tamper detection.

### Event Types

| Event | Description |
|---|---|
| `model_trained` | Training run completed (config + metrics) |
| `model_evaluated` | Evaluation run completed (metrics) |
| `model_deployed` | Model deployed to environment |
| `model_retired` | Model removed from service |
| `data_validated` | Data validation run |
| `drift_detected` | Drift detection triggered |
| `config_changed` | Configuration modified |
| `prediction_flagged` | Prediction flagged for review |
| `human_review` | Human review decision recorded |
| `fairness_evaluated` | Fairness evaluation completed |

### Integrity Verification

Each event's checksum chains with the previous, enabling tamper detection:

```python
from src.governance.audit import audit_trail

# Verify the full audit chain
is_valid = audit_trail.verify_integrity()

# Query recent events
events = audit_trail.get_events(event_type=AuditEventType.MODEL_TRAINED, limit=10)
```

**Storage**: `logs/audit/audit_YYYY-MM.jsonl` (monthly rotation)

Implementation: `src/governance/audit.py`

## Human-in-the-Loop Review

Predictions meeting certain criteria are flagged for clinical review before action.

### Review Triggers

| Trigger | Condition | Priority |
|---|---|---|
| Low confidence | Max prediction probability < 0.7 | medium |
| Conflicting predictions | > 2 predictions borderline (within 0.1 of threshold) | medium |
| High severity | Sight-threatening disease detected (DR, ARMD, CRVO, CRAO, AION, VH, RS) | high |
| Urgent referral | Clinical referral priority = URGENT | urgent |
| Uncertain negative | No diseases detected but > 5 near-threshold predictions | medium |

### API Endpoints

```bash
# Get pending reviews (filtered by priority)
curl http://localhost:8080/api/v1/review/pending?priority=urgent

# Resolve a review
curl -X POST http://localhost:8080/api/v1/review/review_20260425103000/resolve \
  -H "Content-Type: application/json" \
  -d '{"reviewer": "dr.smith", "decision": "confirmed", "notes": "Confirmed DR finding"}'

# Queue statistics
curl http://localhost:8080/api/v1/review/stats
```

### Review Decisions

| Decision | Meaning |
|---|---|
| `confirmed` | Prediction accepted as-is |
| `rejected` | Prediction rejected (false positive) |
| `modified` | Prediction modified by clinician |
| `escalated` | Escalated to specialist |

Implementation: `src/governance/human_review.py`, `backend/app/routers/review.py`

## Key Files

| File | Purpose |
|---|---|
| `src/governance/model_card.py` | Model card generator (JSON + Markdown) |
| `src/governance/dataset_card.py` | Dataset card generator (Gebru et al.) |
| `src/governance/fairness.py` | Fairness evaluator (category + prevalence) |
| `src/governance/audit.py` | Immutable audit trail with integrity verification |
| `src/governance/human_review.py` | Human-in-the-loop review gate |
| `backend/app/routers/review.py` | Review API endpoints |
| `scripts/generate_model_card.py` | CLI for model + dataset card generation |

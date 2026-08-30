# Response to Reviewers — NCC 2026, Submission 83

**Paper:** OptiscanAI: A Benchmark and Deployment Study of Large Vision Models for Multi-Disease
Retinal Screening
*(title revised — see the note under Reviewer 1, Major 1)*

We thank all three reviewers. The revision is substantial: rather than re-describing the system,
we re-ran the evaluation end to end on the **held-out official RFMiD test split (640 images)**
and rebuilt every table from those measurements. Several previously reported numbers did not
survive that re-measurement and have been replaced by what the experiments actually produce; the
specific corrections are listed under Reviewer 1, Major 2 and 6.

Everything in the paper is regenerated from result files by `scripts/ncc2026_tables.py`, so the
manuscript cannot drift from the experiments. The paper is now 8 pages, the NCC 2026 limit for a
full research paper, and uses the official NCC template (IEEEtran, 11 pt) unmodified.

---

## Reviewer 1 (weak reject)

### Major 1 — RFMiD cannot support Uganda-specific clinical claims; frame as benchmark evidence

**Done, including a title change.** "Ugandan Healthcare" is removed from the title, which now
reads *"A Benchmark and Deployment Study."* The abstract closes with "We therefore present these
results as benchmark evidence and a deployment-constraint analysis, not as clinical validation in
Uganda." Section I states the repositioning explicitly.

We also made the gap *quantitative* rather than a caveat. **Table VI** holds the measured
operating point fixed (sensitivity 0.955, specificity 0.731) and varies only prevalence:

| Population | Prevalence | PPV | NPV | Referred |
|---|---|---|---|---|
| RFMiD test (as measured) | 79 % | 0.931 | 0.810 | 81 % |
| Projected at 20 % | 20 % | 0.470 | 0.985 | 41 % |
| Projected at 10 % | 10 % | 0.283 | 0.993 | 34 % |
| Projected at 5 % | 5 % | 0.158 | 0.997 | 30 % |

A programme sized on the benchmark PPV would under-provision its referral pathway by a factor of
three or more (Section V-H).

### Major 2 — Precision-rescue: show the effect on false negatives, recall and sensitivity

**Done — and the re-measurement reversed the previous claim.** Measured on the test split, the
deployed per-class threshold policy is **not** a false-positive reduction. It is recall-first:

| Policy | Sens. | Spec. | PPV | FN | FP | Silent classes |
|---|---|---|---|---|---|---|
| Uniform τ = 0.5 | 0.420 | 0.965 | 0.383 | 199 | 484 | 6 |
| Precision floor (deployed) | 0.591 | 0.720 | 0.114 | 81 | 3 858 | 5 |
| Sensitivity target ≥ 0.90 | 0.811 | 0.729 | 0.189 | 75 | 3 986 | 0 |

It cuts false negatives by 59 % (199 → 81) but raises false positives from 484 to 3 858 and drops
macro PPV from 0.383 to 0.114. The previously reported "22.8 %–37.3 % false-positive reduction"
is not reproducible from the implementation and has been withdrawn.

The reviewer's question also surfaced a **safety defect**: under the shipped policy five disease
channels are assigned the fallback threshold 0.95 and never fire at all, so their sensitivity is
identically zero. A policy that instead targets sensitivity ≥ 0.90 on validation dominates the
deployed one on every axis and leaves no silent classes (**Table III**, new Section V-C).

### Major 3 — Knowledge graph: how it was built, what sources, who validated it

**Done (Section III-D), including a correction and a limitation we now state plainly.**

- **Scale, corrected.** For the 24 deployed classes the graph holds **24 disease nodes and 54
  typed edges**, not the "312 nodes, 1 847 edges" previously reported, which we could not
  reproduce from the implementation.
- **Relation families.** Six: co-occurrence, category membership (vascular, degenerative,
  glaucomatous, infectious/immunologic, neuro-ophthalmic, tractional), three-point severity
  grade, weighted systemic association, age-related risk, and treatment consideration.
- **Sources, recorded per disease.** AAO Preferred Practice Pattern for primary open-angle
  glaucoma (2022) for optic disc cupping; AREDS Report No. 39 for AMD; Uganda Ministry of Health
  diabetes guidelines (2024) for diabetic retinopathy; per-condition journal references for vein
  occlusion and RPE changes, carried in the released graph file.
- **Validation — none yet.** The graph was curated by the authors and has **not** been reviewed
  by an independent ophthalmologist panel. We now say so rather than implying clinical
  endorsement.

### Major 4 — FP32 vs INT8 on the main metrics

**Done (Table IV).** Same 640 test images, identical CPU threading, so accuracy, size and latency
are all comparable:

| Precision | Size | AUC | AUPRC | Sens. | PPV | ms/img |
|---|---|---|---|---|---|---|
| FP32 | 1 224 MB | 0.872 | 0.357 | 0.591 | 0.114 | 434 |
| INT8 dynamic | 310 MB | 0.873 | 0.346 | 0.610 | 0.106 | 418 |
| **Change** | **−74.7 %** | +0.002 | **−0.011** | +0.018 | −0.008 | −4 % |

The reviewer's instinct was right that "negligible loss" needed evidence. It holds for *ranking*
(AUC unchanged) and fails for *retrieval*: AUPRC drops 3.1 % relative, concentrated in rare
classes, and the operating point shifts (8 more true positives, 218 more false alarms). We
therefore recommend re-thresholding after quantisation. We also report the honest negative
result that dynamic quantisation is a **storage** optimisation here, not a speed one — 74.7 %
smaller but only 4 % faster, because the attention matrix products stay in floating point.

### Major 5 and 8 — Per-disease results; AUC alone is insufficient

**Done (Table I).** All 24 retained classes, with positives, prevalence, AUC with 95 % bootstrap
CI, AUPRC, τ_c, sensitivity, specificity, PPV and FNR. Macro AUC is 0.872 (95 % CI 0.846–0.896)
and macro AUPRC 0.357. The spread the aggregate was hiding:

- **Common / strong:** myopia 0.996, media haze 0.974, AMD 0.972, diabetic retinopathy 0.967
  (AUPRC 0.873)
- **Clinically important, mid:** CRVO 0.961, optic disc edema 0.922, BRVO 0.919, optic disc
  cupping 0.783
- **Rare / unreliable:** optic disc pallor 0.663, macular hole 0.674, asteroid hyalosis 0.687

Fig. 2(b) plots AUC against prevalence: every class below roughly 20 test positives has a CI too
wide for a clinical claim, which we now state as a limitation. The false-positive rate the
reviewer lists is reported as its complement, FPR = 1 − specificity, noted under the table so the
column count stays readable.

### Major 6 — Ablation table for LoRA, knowledge graph and quantisation

**Done — one consolidated table (Table II) covering all three**, as asked, plus the two
preprocessing choices from the minor comments. The adaptation arms share a recipe (same data,
loss, 15-epoch schedule, selection on validation mAP, threshold optimisation), so they differ
only in the component named; the post-hoc rows are applied to the deployed checkpoint, so they
are exact rather than re-trained.

| Arm | Params | Trainable | AUC | AUPRC |
|---|---|---|---|---|
| ResNet-50 (ImageNet, full FT) | 23.6 M | 100 % | 0.867 | 0.336 |
| EfficientNet-B3 (ImageNet, FT) | 10.7 M | 100 % | 0.819 | 0.377 |
| ViT-B/16 (ImageNet, full FT) | 85.8 M | 100 % | 0.871 | **0.472** |
| RETFound ViT-L, head only | 305.7 M | 0.8 % | 0.877 | 0.360 |
| RETFound ViT-L + LoRA r=16 | 307.3 M | 1.3 % | **0.882** | 0.392 |
| *— with RFMiD-specific normalisation* | — | — | 0.827 | 0.285 |
| *— with CLAHE* | — | — | 0.866 | 0.341 |
| *+ knowledge-graph reasoning* | — | — | 0.871 | 0.357 |
| *+ INT8 quantisation* | — | — | 0.873 | 0.346 |

This is the item that most changed the paper's conclusions, in both directions:

- **LoRA earns its place:** +0.005 AUC and +0.032 AUPRC over the frozen-head arm for 1.6 M extra
  trainable parameters.
- **But the foundation model does not dominate.** A fully fine-tuned ImageNet ViT-B/16 has the
  best AUPRC of any arm (0.472 vs 0.392). Since AUPRC is the metric that matters under class
  imbalance, we now conclude that retinal pretraining buys **parameter efficiency and ranking**,
  not uniformly better accuracy. The previous draft implied more than this.
- **Knowledge graph:** now measured exactly rather than described. Running the same test
  probabilities through the graph's reasoning rules alters **81 of 640 images (12.7 %) and
  exactly one class (ODE)**, moving macro AUC by **−0.0002** and macro AUPRC by **−0.0000**. Its
  co-occurrence rules almost never fire because most partner conditions they encode are among the
  21 classes dropped for too few training examples. The previously claimed "+3.7 % macro F1 from
  the knowledge graph" is **not supportable and has been withdrawn.** What the graph does
  contribute is measurable and we now report it — see Reviewer 2's third point.
- **Quantisation:** Table IV above.

### Major 7 — Validate the explainability component

**Done (Table V), and it only partially passes.** We ran the deletion/insertion protocol
(Petsiuk et al., BMVC 2018) on 60 confidently detected true positives against a random-saliency
control on the same images:

| Attribution | Deletion AUC ↓ | vs. random | Insertion AUC ↑ | vs. random |
|---|---|---|---|---|
| Grad-CAM | 0.913 | p = 0.029 | 0.926 | p = 0.996 |
| Integrated Gradients | 0.861 | p = 0.008 | 0.943 | p = 0.590 |
| Random (control) | 0.946 | — | 0.943 | — |

Both pass deletion (removing highlighted regions degrades the prediction significantly faster
than removing random ones) and **neither passes insertion**, where both are indistinguishable
from random. We report this rather than presenting the maps as validated, and we downgrade the
claim: they are attention hints requiring clinician verification, not standalone evidence.

For the **natural-language explanations**, we audited 24 generated reports: probability
infidelity 0 % (95 % CI 0–14 %), omission of a detected finding 0 %, and unsupported escalation
of urgency 21 % (95 % CI 9–40 %) — the last being a real weakness we now disclose.

### Minor 1 — ImageNet normalisation may not suit retinal images

**Measured, not argued (Table II, "Input preprocessing").** We re-trained the strongest arm with
RFMiD's own channel statistics in place of ImageNet's. It is **worse**: AUC 0.882 → 0.827,
AUPRC 0.392 → 0.285. The reason is that the backbone was pretrained under ImageNet statistics, so
matching the *pretraining* input distribution matters more than matching the fine-tuning corpus —
the LoRA adapters modify a representation learned in that space. We keep ImageNet statistics and
now state the evidence for it (Section IV).

### Minor 2 — Histogram equalisation could distort clinically meaningful colour

**Resolved twice over.** First, on checking the implementation the deployed pipeline applies **no
histogram equalisation**; the earlier description was simply wrong and has been removed. Second,
because the reviewer's concern was about *effect* rather than presence, we did not leave it as an
assumption: we re-trained the same arm with CLAHE on the L channel of LAB. It **hurts** — AUC
0.882 → 0.866, AUPRC 0.392 → 0.341. That supports the reviewer's reasoning directly: contrast
equalisation is not free on fundus images, and on this corpus it removes more diagnostic signal
than it recovers. Its absence is now a measured choice rather than an oversight (Table II,
Section V-B).

### Minor 3 — Bootstrap CIs promised but not shown

**Fixed.** Table I carries 95 % percentile bootstrap CIs (1 000 image resamples) per class; macro
AUC, macro AUPRC and the referral AUC all carry CIs in the text.

### Minor 4 — Fig. 1 should have numbered components

**Fixed.** Fig. 1 is redrawn with components numbered 1–8, and Section III subsections reference
those numbers directly ("Backbone and Adaptation (2)–(3)").

### Minor 5 — Prototype link returns 404

**Fixed.** The Crane Cloud URL is indeed dead (verified: HTTP 404). It is replaced with the
live deployment at `https://landwind22-retinal-screening.hf.space` (verified serving), labelled
as a workflow demonstration, not a cleared medical device.

---

## Reviewer 2 (accept)

Thank you for the assessment of the systems contribution.

**Limited validation on Ugandan data.** Agreed, and this drove the reframing above: title,
abstract and Section I now present benchmark evidence, and Table VI quantifies the prevalence
transfer. Section VI states plainly that **no Ugandan fundus images were used at any stage**, so
nothing here speaks to local retinal pigmentation, HIV-associated retinopathy, or the handheld and
smartphone-adapted cameras a district health centre would use.

**Quantifying each component's independent contribution.** Table II, under a shared recipe — see
Reviewer 1, Major 6. It changed two of our conclusions.

**Knowledge graph: ontology, sources, expert involvement, validation.** Section III-D — see
Reviewer 1, Major 3, including the correction to 24 nodes / 54 edges and the disclosure that no
independent panel has validated it.

**Evaluation of the explainability results.** Table V — see Reviewer 1, Major 7.

We additionally report the graph's *measurable* contribution, which is referral prioritisation
rather than classification accuracy. Under a non-degenerate threshold policy it stratifies risk
monotonically: 84.5 % of images routed URGENT (86.9 % carry pathology), 10.8 % ROUTINE (47.8 %),
4.7 % FOLLOW-UP (10.0 %). Under the shipped thresholds it labels 100 % of images URGENT and
carries no information at all — a further consequence of the calibration failure below.

---

## Reviewer 3 (accept)

**The false-positive reduction module is not described reproducibly.** Section III-C now gives
the actual mechanism, which differs from the previous description. There is no two-stage cascade
with a secondary classifier; that description did not match the implementation. What exists is
(i) an asymmetric loss with γ_pos = 0, γ_neg = 4, probability clipping 0.05 and label smoothing
0.05, and (ii) a post-hoc per-class threshold fitted on validation as the lowest threshold whose
precision still meets a 0.10 floor, with τ_c = 0.95 as a fallback. Both are stated with their
hyperparameters, and Table III measures what the policy costs.

**Integration of the GAT module into the classifier is not explained.** This was the most
valuable single comment, because on inspection **there is no GAT in the classifier.** The
previous description of a graph embedding fused with the vision embedding by cross-attention did
not match the code. What the network contains is a single sparse top-k attention layer (k = 32)
over the ViT patch tokens — patch-level self-attention, not a graph attention network over the
clinical knowledge graph. The knowledge graph is a **post-hoc symbolic layer** applied after
classification. Section III-B now says this explicitly, including the negative statement, and
Fig. 1 places the graph after the classifier accordingly.

**Generalisability to the local population and imaging conditions cannot be assumed.** Agreed,
and we no longer paper over it — see the Reviewer 2 response and Table VI.

---

## Additional finding, not raised by the reviewers

Chasing Reviewer 1's Major 2 uncovered the root cause of several symptoms: the model is severely
**over-confident**. Mean predicted probability is 0.251 against an empirical positive rate of
0.042, the minimum probability anywhere on the test split is 0.064, and expected calibration
error is 0.209. Because nothing is ever confidently negative, the precision-floor search drives
thresholds to the bottom of the grid for common classes (specificity → 0) and to the fallback for
rare ones (sensitivity → 0).

This one fact explains the degenerate thresholds, the five silent classes, the uninformative
referral prioritisation, and the flat insertion curves in the explainability audit. Per-class
Platt scaling fitted on validation reduces test ECE from 0.209 to **0.003** while leaving AUC
unchanged at 0.872, cuts false positives from 3 986 to 2 356 and raises macro PPV from 0.189 to
0.216 (Section V-D, Fig. 2(a), Table III). We attribute the miscalibration to the asymmetric
loss: clipping at 0.05 removes the gradient that would push confident negatives towards zero.

---

## Alignment with current reporting standards

The revision is now written against the guidance that governs this kind of study, which the
earlier version did not reference:

- **CLAIM 2024 update** (Tejani et al., *Radiology: AI* 6(4):e240300) — medical-imaging AI
  reporting: single held-out split used once, per-outcome results, interval estimates.
- **TRIPOD+AI** (Collins et al., *BMJ* 385:e078378, 2024) — prediction-model reporting:
  discrimination **and** calibration, explicit intended population.
- **FUTURE-AI** (Lekadir et al., *BMJ* 388:e081554, 2025) — six principles for deployable
  healthcare AI. We map the work onto them honestly: this study provides evidence for
  **traceability**, **explainability** (including where it fails) and **robustness** under
  quantisation, and provides **no** evidence for **fairness** or **universality**, because those
  require the demographically annotated local data we do not have.

This is added as a "Reporting standards" paragraph in Section IV.

## Changes carried back into the system, not just the paper

The findings above are defects in the deployed configuration, so we fixed them in the codebase
rather than only describing them:

1. **Calibration is now part of the model.** `RetinalFoundationHybridV2` carries per-class Platt
   parameters as buffers and applies them on both the single-pass and test-time-augmentation
   inference paths. They default to the identity, so an existing checkpoint behaves exactly as
   before, and `load_calibration()` installs a fitted calibration together with the thresholds
   fitted against it — the two travel as a pair, because either alone puts the model at an
   operating point neither was chosen for. Six new tests cover this (27 pass, no regressions).
2. **Wrong disease names corrected.** The label glosses had **DN as "Diabetic Neuropathy"** (it is
   *drusen*) and **MH as "Macular Hole"** (it is *media haze*), among others — these reach
   clinician-facing text. 15 labels corrected across `model_explainer.py` and
   `backend/app/core/model_service.py` against the RFMiD definitions.

## Summary of corrections to previously reported results

For transparency, these figures from the accepted version were not reproducible from the
implementation and have been replaced by measured values:

| Previously reported | Status | Replaced by |
|---|---|---|
| Macro F1 0.507, AUC–ROC 0.856 | Not reproducible | Macro AUC 0.872 (95 % CI 0.846–0.896), macro AUPRC 0.357, per-disease Table I |
| FPR reduction 22.8 %–37.3 % from precision rescue | Direction was wrong | FN −59 % at the cost of FP 484 → 3 858 (Table III) |
| Knowledge graph +3.7 % macro F1, +8.3 % consistency | Not supportable | Graph contributes referral stratification, not accuracy |
| Knowledge graph 312 nodes, 1 847 edges | Not reproducible | 24 nodes, 54 typed edges for the 24 deployed classes |
| Two-stage cascade with secondary classifier | Not in the implementation | Asymmetric loss + per-class threshold policy (Section III-C) |
| GAT embedding fused by cross-attention | Not in the implementation | Sparse top-k patch self-attention; graph applied post-hoc (Section III-B, III-D) |
| 512 × 512 inputs, histogram equalisation | Not in the implementation | 224 × 224 bicubic, ImageNet normalisation, no equalisation (Section IV) |
| Quantisation 342 MB → 87 MB (74.6 %) | Absolute sizes wrong | 1 224 MB → 310 MB (−74.7 %); student 20.8 → 5.4 MB (−73.8 %) |
| Model size / dataset framing "3 200 images, 70/15/15" | Split misdescribed | Official RFMiD partition 1 920 / 640 / 640 |

## Reproducibility

Every table is generated from result files by `scripts/ncc2026_tables.py`. The pipeline is:
`ncc2026_infer.py` (inference) → `ncc2026_metrics.py` (per-disease metrics with bootstrap CIs) →
`ncc2026_calibration.py`, `ncc2026_operating_points.py`, `ncc2026_referral.py`,
`ncc2026_explain_eval.py`, `ncc2026_train_arm.py` (ablation arms) → `ncc2026_tables.py` and
`ncc2026_figures.py`.

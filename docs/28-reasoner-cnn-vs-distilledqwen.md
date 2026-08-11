# Replacing the external LLM reasoner: CNN vs DistilledQwen

> Status: **plan + harness + smoke run + REAL run all delivered** (real run
> executed on GPU against the live Qwen3-8B teacher — see
> [§0 Real-data results](#0-real-data-results-executed)). No Docker image is built
> until the result is reviewed and signed off.
>
> ✅ **Shipped (triage half).** The learned triage head is wired into the
> pipeline and baked into the Docker images: `src/triage/` serves
> `models/triage/triage_model.json` (3 KB of JSON weights — no pickle, no
> scikit-learn at serve time) from `triage_node`, ahead of the LLM, with a
> deterministic emergency override and a rules fallback. Toggle with
> `TRIAGE_MODEL_ENABLED` / `TRIAGE_MODEL_PATH` (see
> [doc 24](24-environment-variables.md)). The exporter refuses to write weights
> that disagree with the fitted model, and the loader refuses an artifact whose
> feature layout has drifted.
>
> ✅ **Shipped (narrative half, opt-in).** A compact prose-only narrator
> (SmolLM2-135M, vocabulary pruned 49,152 → 920) is published at
> [`Mpairwe49/retinalai-narrator-135m`](https://huggingface.co/Mpairwe49/retinalai-narrator-135m)
> and wired into `report_node` via `src/narrator/`. It is **off by default**
> (`NARRATOR_ENABLED=false`) because it has not been clinically reviewed and was
> scored on 24 cases; the grounded template remains the default. Only **bf16** is
> recommended — 4-bit reaches 54 MB but drops reported findings in 42% of cases.
> The AI-disclosure sentence is appended in code, since no teacher trace contains
> one. See [doc 29 §3.11 and §3.13](29-narrator-verification-and-gaps.md).
>
> ⚠️ **Read [`29-narrator-verification-and-gaps.md`](29-narrator-verification-and-gaps.md)
> before acting on §0.6–§0.9.** A review of the narrator work found the
> generation-rate comparison confounded by a 200-token cap (~40% of teacher
> targets were truncated mid-JSON), the p50/p95 latency figures non-comparative
> (all narrators measure ~11 tok/s regardless of size, 25–102× below their
> memory roofline), and `grounding` blind to invented clinical severity. The
> §0.9 narrator pick is **downgraded to "candidate pending re-measurement"**.
> Triage conclusions (§0.5/§0.8) are unaffected.

## TL;DR

The screening pipeline delegates two jobs to an external LLM (the aspirational
self-hosted `Qwen/Qwen3-8B-AWQ`, today stood in for by Claude → Groq →
deterministic rules): **(1) structured triage** and **(2) a free-text clinical
narrative**. "Replace Qwen with a CNN" and "distill Qwen" are *not* the same swap
because a CNN can do (1) but **cannot** do (2). This doc defines a head-to-head
that scores both candidates on the *same* task, ships a runnable harness, and a
smoke run that already surfaces the key risk: **a learned triage model that
misses an EMERGENCY fails the safety gate** even when its overall F1 looks fine.

**Recommendation:** the **triage** half is settled — a **2 KB
logistic-regression head** reproduces the live Qwen3-8B teacher's triage
*exactly* (acc/macro-F1/macro-precision 1.000, held-out, 5-fold CV, and on 160
fresh out-of-sample images), beating the image CNN (0.520 / 0.492) at **2 KB and
sub-millisecond** inference; deterministic rules remain the zero-dependency
floor. Ship that.

The **narrator** half is **not settled**. §0.7 selected a server-hosted
DistilledQwen-0.5B (bf16, 988 MB); §0.9 proposed swapping it for
SmolLM2-360M-Instruct (724 MB, −27%). Review
([doc 29](29-narrator-verification-and-gaps.md)) found the narrator evidence
does not support a pick: generation-rate differences were an artifact of a
200-token cap that truncated ~40% of teacher-length targets, the latency figures
measure harness overhead rather than the models (all narrators clock ~11 tok/s
regardless of size), and `grounding` is blind to invented clinical severity —
the proposed model is the one that fabricates *"life-threatening"* and
*"permanent vision loss"*. The grounded template (0 MB) remains the safe
zero-compute default until a narrator is re-qualified under §29's roadmap
(constrained decoding, a real serving stack, claim-level faithfulness eval,
and more than 80 traces).

---

## 0. Real-data results (executed)

Run executed on a free GPU against the **live self-hosted Qwen3-8B-AWQ** (vLLM
@ `localhost:8011`) as teacher. 80 real RFMiD fundus images → `model_service`
predictions → teacher triage+narrative traces (`traces.jsonl`) → LoRA-distilled
Qwen2.5-0.5B narrator (`outputs/distilled_qwen/`) → 4-way comparison on a held-out
split (test n=24). Full table: [`outputs/reasoner_comparison_real/report.md`].

| reasoner | triage macro-F1 | triage acc | κ vs teacher | narrative | grounding | words | size MB | p95 ms | gate |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|:--:|
| `rule_baseline` | **1.000** | 1.000 | 1.000 | template | 1.000 | 34 | 0.0 | 0.0 | ✅ **PASS** |
| `cnn_triage` | 0.520 | 0.792 | 0.429 | template | 1.000 | 34 | 6.1 | 161 | ❌ FAIL (F1<0.70) |
| `distilled_qwen` | **1.000** | 1.000 | 1.000 | ✅ generative | 1.000 | 66 | 1976 | 6579 | ❌ FAIL (size+latency) |
| `qwen_teacher` | 1.000 | 1.000 | — (ref) | ✅ generative | 1.000 | 66 | — | — | reference |

**What the numbers say:**

1. **Rules already are the teacher, for triage.** On real data the deterministic
   rule mapping reproduces the 8B teacher's 4-class priority *exactly* (macro-F1
   and κ = 1.000). There is **no learned triage model to justify** — a CNN or a
   distilled LLM can at best tie the free baseline.
2. **The CNN is strictly worse and fails the gate.** Image→triage MobileNetV3
   lands at macro-F1 0.520 (κ 0.429) — it *underperforms* the rules it was meant
   to replace and trips the `min_priority_macro_f1` 0.70 gate. The CNN learns the
   image→label mapping less reliably than the rules read it straight from the
   classifier's probabilities.
3. **Only DistilledQwen reproduces the narrative offline — but it's too heavy.**
   The 0.5B distilled narrator matches the teacher's triage (1.000) *and* its
   prose (grounding 1.000, 66 words, near-identical phrasing — see sample below),
   confirming distillation works. But at **1976 MB / 6.6 s p95** it blows the
   `max_size_mb` 60 and `max_latency_p95_ms` 1800 edge gates by ~33× and ~3.7×.
   It is shippable *only* after INT4 quantization + a token cap, which is a
   separate piece of work.
4. **The flags are uninformative on this data.** `should_explain` / `should_review`
   are all-True across every reasoner (the teacher always sets them for any
   positive finding), so they carry no discriminative signal here.
5. **No EMERGENCY cases in the sample.** RFMiD-24 + the current over-detecting
   classifier (threshold 0.53) yields a triage spread of URGENT 56 / FOLLOW_UP 14
   / ROUTINE 10 over 80 traces and **zero EMERGENCY** in the test split, so the
   `emergency_recall = 1.0` gate is *vacuously* satisfied — it is **not** evidence
   the learned models are emergency-safe. That gate stays unproven until a sample
   with real EMERGENCY cases is run.

Near-identical teacher vs. distilled narrative on the first test case (distillation
fidelity):

> **teacher** — "Optic Disc Pallor (60%), Laser Scars (58%), Diabetic Retinopathy
> (56%), and Central Serous Retinopathy (56%) were detected in the retinal scan.
> These findings indicate multiple significant retinal pathologies with medium
> confidence levels. …"
>
> **distilled_qwen** — "Optic Disc Pallor (60%), Laser Scars (58%), Diabetic
> Retinopathy (56%), and Central Serous Retinopathy (56%) were detected in the
> retinal scan. These findings suggest multiple medium-confidence retinal
> pathologies with potential for rapid v…"

**Decision:** triage → a **2 KB learned model matches the 8B teacher exactly**
(see §0.5) and is preferred over the CNN; rules remain the zero-dependency floor.
Narrative → **ship the grounded template now**; the DistilledQwen narrator, even
4-bit quantized, stays too heavy for the edge (§0.6). The Docker image needs
**no LLM**.

> **Superseded for the narrative half.** The paragraph above was written while
> the *edge* size/latency gate still applied. §0.7 lifted that gate (hosting is
> server-side) and selected a server-hosted narrator; §0.9 revisited the base
> model. The template remains the correct choice only for a true edge/no-GPU
> deployment. Triage half is unchanged and still current.

### 0.5 Follow-up — lightweight learned triage (other architectures)

The CNN's failure was diagnostic: it predicts triage from *pixels*, but the
teacher derives it from the classifier's **structured output** (detected diseases
+ probabilities + referral). Feeding a tiny tabular model that structured vector
(`features.case_features`, via `scripts/sweep_triage_architectures.py`) recovers
the teacher's mapping outright. Same held-out split as the CNN, plus 5-fold CV for
generalizability:

| triage model | hold acc | hold macro-F1 | hold macro-P | 5-fold CV acc | size | p95 | gate |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `cnn_triage` (pixels) | 0.792 | 0.520 | 0.492 | — | 6.1 MB | 161 ms | ❌ |
| **`feat_logreg`** | **1.000** | **1.000** | **1.000** | **1.000** | **2 KB** | 0.14 ms | ✅ |
| `feat_decision_tree` | 1.000 | 1.000 | 1.000 | 1.000 | 2 KB | 0.17 ms | ✅ |
| `feat_mlp` | 1.000 | 1.000 | 1.000 | 1.000 | 0.14 MB | 0.24 ms | ✅ |
| `feat_random_forest` | 1.000 | 1.000 | 1.000 | 1.000 | 0.25 MB | 11 ms | ✅ |
| `feat_xgboost` | 1.000 | 1.000 | 1.000 | 1.000 | 0.42 MB | 1.8 ms | ✅ |
| `rule_baseline` (ref) | 1.000 | 1.000 | 1.000 | — | 0 | 0.02 ms | ✅ |

Five of seven architectures hit a perfect 1.000 on accuracy **and** macro-precision,
held-out and under cross-validation — clearing the "≥75% accuracy, ≥ the CNN's
precision" bar by the maximum margin, at KB scale and sub-millisecond latency. An
emergency code deterministically escalates to EMERGENCY regardless of the learned
head, so the safety floor is preserved by construction. **Best pick:
`feat_logreg`** — 2 KB, 0.14 ms, saved to `outputs/triage_model/`.

A **findings-only ablation** (drop the referral feature) still keeps decision-tree,
MLP and XGBoost at 1.000 held-out, so the model genuinely learns the
findings→triage mapping rather than parroting the classifier's referral. *Honest
caveat:* on this RFMiD sample the teacher's triage essentially equals the
classifier's referral passthrough (no EMERGENCY fired, §0.5), which is why the task
is so cleanly separable — the next step is to confirm this holds on a more varied
sample (see §10), which is exactly the generalizability check that precedes any
Docker build.

### 0.6 Follow-up — 4-bit quantizing the narrator (bitsandbytes NF4)

`scripts/quantize_distilled_reasoner.py` loads the 0.5B narrator with the reliable
HF-integrated **NF4 double-quant** path and re-scores it on the held-out split:

| narrator | size MB | p50 ms | p95 ms | grounding | gate |
|---|---:|---:|---:|---:|:--:|
| fp baseline | 1976 | 5152 | 6371 | 1.000 | ❌ |
| **NF4 4-bit** | **451** | 10850 | **12923** | 1.000 | ❌ |

4-bit cuts the footprint **4.4×** (1976 → 451 MB) with grounding fully preserved —
but it is **still 7.5× over the 60 MB edge gate**, and on a 0.5B model it ran
**~2× slower**: bitsandbytes 4-bit trades VRAM for per-token dequant overhead, a
win for fitting large models in limited memory, not for speeding up a small one.
**Quantization alone does not make the narrator edge-deployable.** No generative
0.5B model meets a 60 MB budget (the 152K-vocab embedding alone is ~272 MB).

### 0.7 Decision — host the narrator on a **server** (edge gate removed)

Because the narrator is hosted in the backend, not on a phone, the size/latency
*edge* gate does not apply. The **server gate** (`metrics.SERVER_GATES`) keeps the
safety + quality checks — emergency recall, triage macro-F1, narrative grounding —
and drops the artifact-size and mobile-latency caps. Re-measured across precisions
(`scripts/quantize_distilled_reasoner.py`, 24-case held-out split):

| narrator (server) | size MB | p50 ms | p95 ms | grounding | server gate | edge gate |
|---|---:|---:|---:|---:|:--:|:--:|
| fp32 | 1976 | 4440 | 5352 | 1.000 | ✅ PASS | ❌ |
| **bf16 (recommended)** | **988** | 5888 | 6935 | 1.000 | ✅ PASS | ❌ |
| NF4 4-bit | 451 | 10720 | 12999 | 1.000 | ✅ PASS | ❌ |

All three **pass the server gate** with grounding 1.000 and faithful teacher-style
prose. Selection for "lightweight + reasonable compute":

- **bf16 on GPU is the pick** — it halves fp32's footprint (1976 → 988 MB) at
  identical quality. fp32 measured marginally faster here only because batch-1,
  short-generation decoding is launch/Python-overhead-bound, not compute-bound, so
  dtype doesn't change throughput; that small gap is within the contention noise of
  this shared box. **NF4 4-bit** (451 MB) is the choice *only* if GPU VRAM is the
  binding constraint — it is smallest but ~2× slower.
- *Latency caveat:* the 5–13 s/case figures are inflated by heavy co-tenant load on
  this host (ship-pretrain + vLLM). On a dedicated server GPU a 0.5B narrator
  generating ~60 words is typically **~1–2 s/case**.

**Final server stack (best option under the constraints at the time):**

| sub-task | model | size | quality | server gate |
|---|---|---:|---|:--:|
| triage | `feat_logreg` (2 KB structured-feature LR) | 2 KB | acc/precision 1.000 (= 8B teacher), CV 1.000 | ✅ |
| narrative | `distilled_qwen` 0.5B, **bf16 on GPU** | 988 MB | grounding 1.000, reproduces teacher prose | ✅ |

This pair satisfies the brief — **>75% accuracy with ≥ the CNN's precision** (in
fact a perfect 1.000 on both), a **lightweight, low-compute** triage head, and a
**reasonable-quality generative narrative** — now that hosting is server-side. The
grounded template (0 MB) remains the zero-compute fallback if the narrator is
disabled.

> **Superseded:** §0.9 found a smaller base model (SmolLM2-360M-Instruct) that
> matches this narrator on every scored metric at 27% less size — see §0.9 for
> the updated pick. Triage row is unchanged.

### 0.8 Generalizability test (the pre-Docker gate) — accuracy + precision

`scripts/generalizability_test.py` validates the triage head beyond the single
24-case split, on four axes. Fresh teacher traces were generated over **160
disjoint, previously-unseen RFMiD images** (train = original 80; the OOS set has a
*different* label mix — URGENT 106 / ROUTINE 32 / FOLLOW_UP 22 vs train's 56/10/14).

| test | metric | result |
|---|---|---|
| **True out-of-sample** (train 80 → test 160 fresh) | acc / macro-P / F1 / κ | **1.000 / 1.000 / 1.000 / 1.000** (logreg, tree, MLP, XGBoost) |
| **Repeated 5-fold CV × 5 seeds** (25 fits) | acc, macro-P (mean ± std) | **1.000 ± 0.000 / 1.000 ± 0.000** |
| **Learning curve** (logreg, OOS) | acc / macro-P at n=20…80 | **1.000 / 1.000** from just **20** train cases |
| **EMERGENCY stress** (80 injected CRAO/AION) | recall / precision / F1 | **1.000 / 1.000 / 1.000** |

The triage head **generalizes perfectly and is emergency-safe by construction**.
But the decisive diagnostic is *why*: on the OOS set the **teacher's triage equals
the classifier's `referral_priority` in 160/160 cases (zero divergences)**. The 8B
teacher adds **nothing** to triage beyond the referral the classifier already
emits — so the task is deterministic, which is why a 2 KB linear model (or the
rules, or the 8B) all score an identical 1.000. The flat learning curve from n=20
confirms it.

**What this means for deployment:**

1. **Triage is solved and validated** — ship the 2 KB `feat_logreg` (an
   emergency-safe re-encoding of the referral) or the rules; neither a CNN, a
   distilled LLM, nor the 8B is needed for the *decision*.
2. **The LLM's real value is the narrative** — that is genuine generation the
   classifier cannot produce (the server-hosted bf16 narrator, §0.7).
3. **Residual risk moved upstream.** No real EMERGENCY case appeared in 240 RFMiD
   images (0 CRAO/AION detected), so emergency behavior is verified only via the
   injected stress test (the override guarantees escalation). Whether real
   emergencies are *caught* depends on the **disease classifier** detecting CRAO/
   AION — a classifier-quality question, outside this reasoner-replacement scope,
   and the thing to probe before clinical use.

### 0.9 Follow-up — narrator base-model architecture sweep

§0.5 tried other architectures for the *triage* head; this does the same for the
*narrator*. The current pick (Qwen2.5-0.5B-Instruct, §0.7) is 988 MB in bf16
because its tokenizer has a 151936-token vocabulary — the embedding table alone
is ~272 MB, independent of how "small" the model nominally is. HuggingFaceTB's
SmolLM2-Instruct family uses the same LoRA-SFT recipe
(`distill_qwen_reasoner.py --base-model ...`, same 56-row training set) but a
49152-token vocabulary, so it was distilled at two sizes and re-scored on the
identical 24-case held-out split (`scripts/sweep_narrator_architectures.py` →
[`outputs/reasoner_comparison_real/narrator_arch_sweep.md`]):

All six variants re-measured through one code path at a **384-token** budget
(the original 200-token cap truncated ~40% of teacher-length targets mid-JSON):

| narrator | size MB | p95 ms † | grounding | **gen rate (95% CI)** | gate (edge / server) |
|---|---:|---:|---:|---:|:--:|
| `qwen0.5b_bf16` (§0.7 pick) | 988.1 | 7915 | 1.000 | 1.000 [0.86, 1.00] | ❌ / ✅ |
| **`qwen0.5b_nf4`** | **451.3** | 15557 | **1.000** | **1.000** [0.86, 1.00] | ❌ / ✅ |
| `smollm2_360m_bf16` | 723.6 | 7081 | 1.000 | 1.000 [0.86, 1.00] | ❌ / ✅ |
| `smollm2_360m_nf4` | 251.8 | 13793 | 0.972 | 0.958 [0.80, 0.99] | ❌ / ✅ |
| `smollm2_135m_bf16` | 269.0 | 7003 | 1.000 | 0.917 [0.74, 0.98] | ❌ / ✅ |
| `smollm2_135m_nf4` | 109.8 | 28622 | 1.000 | **0.000** [0.00, 0.14] | ❌ / ✅ |

† **Latency is not comparative.** All bf16 variants clock ~11 tok/s despite a
3.7× spread in weight volume — the figures measure harness overhead, not the
models ([doc 29 §2.3](29-narrator-verification-and-gaps.md)).

`gen rate` is new: the fraction of cases where the model's own JSON output
actually contained a usable `narrative` field, vs. silently falling back to the
grounded template. This distinction wasn't tracked before this sweep — and it
turned out to matter: the template is grounded *by construction*, so
**grounding alone can't tell a genuinely generative narrator from one that is
quietly failing on every case** (`ReasonerOutput.narrative_generated`, plumbed
through in this change; a related crash was also fixed — a degraded model can
emit valid JSON that parses to a bare string rather than an object, which
`_extract_json` didn't previously guard against).

**Findings:**

1. **`smollm2_360m_bf16` is the leading candidate — but "strictly beats" was
   overstated.** It matches on grounding (1.000) with a 100% generation rate at
   **724 MB vs. 988 MB — 27% smaller** for the same recipe and data. However
   (see [doc 29 §2](29-narrator-verification-and-gaps.md)): the Qwen comparison
   figure was *asserted, not measured* — measured directly, Qwen0.5B scores
   0.833 at the 200-token budget this sweep used and 1.000 at an adequate 384,
   so the two are **equal on reliability once the truncation bug is fixed**, and
   the honest remaining margin is size alone. The latency "tie" is not a finding:
   all narrators here measure ~11 tok/s regardless of size, so the latency axis
   is measuring harness overhead, not the models.
2. **`smollm2_135m_bf16` is smaller (269 MB, 3.7× less than Qwen) but less
   reliable** — 91.7% generation rate, i.e. roughly 1 in 12 cases silently
   degrades to the template. Not a strict win the way the 360M pick is; a
   candidate for future work if the reliability gap can be closed (more
   epochs/training data — this sweep reused the exact §0.7 recipe unchanged).
3. **NF4 damage is capacity-dependent — "4-bit is unsafe" was wrong in both
   directions.** Re-measured at an adequate token budget:

   | base model | params | NF4 gen rate |
   |---|---:|---:|
   | Qwen2.5-0.5B | 494M | **1.000** — no degradation whatsoever |
   | SmolLM2-360M | 362M | 0.958 (grounding also slips to 0.972) |
   | SmolLM2-135M | 135M | **0.000** — destroyed |

   Smaller models carry less redundancy, so identical quantization error does far
   more damage. The 135M collapse is real (a 512-token probe with zero truncation
   still parsed 0/6), but NF4 costs the 0.5B model *nothing*. Note also this is
   **bitsandbytes**-specific — a dev-convenience path, not a serving quantizer;
   this project's own teacher is `Qwen3-8B-AWQ` (4-bit, vLLM, working), and
   AWQ/GPTQ narrator variants remain untested ([doc 29 §3.2](29-narrator-verification-and-gaps.md)).

   **This surfaces an option the sweep's framing missed:** `qwen0.5b_nf4` is
   **451 MB** at grounding 1.000 / generation 1.000 — *smaller than
   `smollm2_360m_bf16` (724 MB) at equal reliability*. Asking "which base model
   at bf16" hid the fact that quantizing the incumbent beats switching base
   models on size. Its faithfulness is unaudited.

   Separately, grammar-constrained decoding (XGrammar/Outlines) makes malformed
   JSON structurally impossible and would likely eliminate this failure mode
   entirely — it should be tried before any model is eliminated on reliability
   ([doc 29 §3.1](29-narrator-verification-and-gaps.md)).
   Given DEFAULT_GATES/SERVER_GATES both score grounding off the *final*
   narrative (template-backstopped), this failure mode was invisible until
   generation rate was tracked explicitly — worth carrying forward as a
   standing check on any future quantized-narrator change.
4. **No generative candidate clears the 60 MB / 1800 ms edge gate.** Closest
   is `smollm2_135m_nf4` at 109.8 MB, but it is functionally non-generative
   (0% generation rate). The edge gate remains template-only / no-narrator;
   this sweep narrows the gap but doesn't close it.

**Updated pick — WITHDRAWN pending re-measurement.** The original conclusion here
was to swap the server-hosted narrator to `distilled_smollm2_360m` (724 MB vs
988 MB). Review ([doc 29](29-narrator-verification-and-gaps.md)) found the
supporting evidence insufficient:

- the reliability advantage was an artifact of the 200-token cap — at an
  adequate budget **both** models generate on 24/24 cases;
- the latency "tie" measured harness overhead, not the models;
- a claim-level faithfulness audit found `smollm2_360m` is the **only**
  candidate that invents hard severity language (*"life-threatening"*,
  *"permanent vision loss"* — absent from all 80 teacher traces), while
  `smollm2_135m` omits reported findings in 13.6% of cases. All scored
  `grounding = 1.000`.

What survives: the **size** finding (724 MB vs 988 MB at equal grounding and
generation rate) is real, and SmolLM2-360M remains the leading *candidate*.
It is not yet a justified swap. Triage head (§0.5/§0.8) is unaffected.

---

## 1. What "Qwen" actually does here (analysis)

The reasoner is invoked at exactly two nodes of the LangGraph pipeline
(`src/agents/graph.py`), both operating on the **classifier's disease
predictions**, never on raw pixels:

| Node | Input | Output | Type | Fallback today |
|---|---|---|---|---|
| `triage_node` (`graph.py:140`) | detected diseases + probs + classifier referral | `priority` ∈ {EMERGENCY, URGENT, ROUTINE, FOLLOW_UP}, `should_explain`, `should_review`, one-line `reasoning` | **structured** | `_rule_based_reasoning` (`graph.py:208`) |
| `report_node` (`graph.py:314`) | detected diseases + triage | 3–4 sentence clinical `narrative` | **free text** | `_template_narrative` (`graph.py:390`) |

Two findings that shape everything:

1. **The vLLM `Qwen3-8B-AWQ` endpoint is not wired into any live Python path.**
   It exists only in runbooks + a config flag `quantization.vllm_enabled`
   (default `False`). The live LLM is Claude → Groq → deterministic rules, and
   the `reason_node` is *already* deterministic (knowledge-graph). So "replacing
   Qwen" means **replacing the external-LLM reasoner *role*** with a
   self-contained component — not deleting a wired dependency. The production
   Docker image already ships **no LLM** (it bakes the ViGNN classifier +
   fundus gate and calls out / falls back).

2. **CNN ≠ LLM, by construction.** Triage is a 4-way classification a CNN/MLP can
   learn; the narrative is generation only an LLM-style model can do. A fair
   comparison must therefore score **two sub-tasks separately** and be explicit
   that the CNN scores *zero* on genuine narrative.

## 2. The candidates

| Candidate | What it is | Triage | Narrative | Offline | New deps |
|---|---|:--:|:--:|:--:|---|
| `rule_baseline` | Existing deterministic rules + template (`graph.py`) | ✅ | template | ✅ | none |
| `cnn_triage` | **The "CNN"**: MobileNetV3-Small → triage heads (image → decision) | ✅ | template | ✅ | none (timm already in repo) |
| `distilled_qwen` | **The "DistilledQwen"**: a small local causal LLM (e.g. Qwen2.5-0.5B/1.5B-Instruct) fine-tuned/distilled on teacher traces | ✅ | ✅ generative | ✅ | `transformers`, `accelerate` |
| `llm_teacher` | Production LLM (Claude/Groq, or a self-hosted Qwen) — the **oracle / upper bound** and the source of training labels | ✅ | ✅ generative | ❌ | provider SDK |

MobileNetV3-Small is deliberate: it shares the operator set of the existing
fundus gate / mobile student, so the ONNX/INT8 export path is already solved
(cf. `src/models/mobile_student.py`).

## 3. The common task & metrics

Scored against the **teacher** as reference (`src/evaluation/reasoner_comparison/metrics.py`):

- **Triage (structured):** priority accuracy, macro-F1 (over observed labels),
  per-class P/R, **EMERGENCY recall** (safety-critical), Cohen's κ vs teacher,
  and `should_explain` / `should_review` F1.
- **Narrative (free text):** **grounding** (fraction of disease mentions that
  were actually detected — 1.0 = no hallucinated diseases), top-finding
  coverage, empty-rate, avg length. A CNN's templated text scores "safe but
  generic"; only a real LLM moves these meaningfully.
- **Ops:** size (MB), p50/p95 latency, offline?, extra deps.

### Deployment gates (`DEFAULT_GATES`)

| Gate | Threshold | Rationale |
|---|---|---|
| EMERGENCY recall | **= 1.0** (when emergencies present) | Missing a sight-threatening case is unacceptable |
| priority macro-F1 | ≥ 0.70 | Triage must be usefully better than chance |
| size | ≤ 60 MB | Edge / Docker-image budget |
| p95 latency | ≤ 1800 ms | Mid-range Android target (mirrors mobile student) |
| narrative grounding | ≥ 0.95 *(narrative-capable only)* | No hallucinated diseases in reports |

## 4. The harness

`src/evaluation/reasoner_comparison/` — every candidate implements one
`Reasoner` interface so they're interchangeable:

| File | Role |
|---|---|
| `interface.py` | `Case`, `ReasonerOutput`, `Reasoner` ABC, priority/critical-code constants (mirror `graph.py`) |
| `reasoners.py` | `RuleReasoner`, `CNNTriageReasoner`, `DistilledLLMReasoner`, `LLMReasoner` |
| `cnn.py` | `TriageCNN` (MobileNetV3-Small + 3 heads) + `train_triage_cnn` |
| `cases.py` | synthetic generator (smoke) + real RFMiD loader (gated) |
| `metrics.py` | triage / narrative / ops scorers + gate evaluation |
| `runner.py` | run all reasoners → `results.json` + `report.md` |

Scripts:
- `scripts/run_reasoner_comparison.py --mode {smoke,real}` — the entry point.
- `scripts/generate_reasoning_traces.py --mode {synthetic,llm}` — builds the
  teacher-trace dataset (§5).
- `scripts/build_sft_dataset.py` — turns traces into the narrator's SFT set.
- `scripts/distill_qwen_reasoner.py --base-model ...` — LoRA-distills any HF
  causal LM (Qwen2.5-0.5B-Instruct, SmolLM2-*-Instruct, …) on the SFT set.
- `scripts/sweep_triage_architectures.py` — other architectures for the triage
  head (§0.5).
- `scripts/quantize_distilled_reasoner.py` — precision sweep (fp32/bf16/NF4) on
  one narrator (§0.6/§0.7).
- `scripts/sweep_narrator_architectures.py` — base-model sweep across narrators
  (§0.9).
- `scripts/generalizability_test.py` — OOS/CV/learning-curve/emergency-stress
  checks on the triage head (§0.8).

Tests: `tests/test_reasoner_comparison.py` (29 tests; CNN/runner cases marked
`slow`).

```bash
# Offline, no GPU/keys — validates the whole harness in ~20s:
PYTHONPATH=. python scripts/run_reasoner_comparison.py --mode smoke
PYTHONPATH=. python -m pytest tests/test_reasoner_comparison.py -q
```

## 5. The teacher-trace dataset (the net-new piece)

Both learned candidates need a **teacher signal** per case: the
priority/flags/reasoning + narrative the production LLM would produce. This does
not exist in the repo yet. `scripts/generate_reasoning_traces.py`:

- `--mode llm` runs the teacher LLM over RFMiD predictions → JSONL labels. This
  is the **API-cost / time** step quantified in §8.
- `--mode synthetic` emits labels from a deterministic stand-in teacher — for
  wiring/testing only, **not** a substitute for real labels.

RFMiD has ~3,200 images; one teacher call-pair (triage + report) per image.

## 6. Smoke-run results (harness validation only)

From `outputs/reasoner_comparison/report.md` — **synthetic** cases, CNN
micro-trained 20 epochs on CPU. These validate the *harness and learnability*,
**not** production quality (the synthetic teacher is largely rule-derived, which
is why the rule baseline looks strong):

| reasoner | macro-F1 | EMERG recall | κ | size | p95 | narrates | gate |
|---|---:|---:|---:|---:|---:|:--:|:--:|
| `rule_baseline` | 0.907 | 1.000 | 0.888 | 0.0 MB | ~0 ms | template | ✅ PASS |
| `cnn_triage` | 0.733 | **0.800** | 0.663 | 6.1 MB | 9.5 ms | template | ❌ **FAIL** |

**What this already tells us:**

1. The harness runs end-to-end, the CNN genuinely learns (loss 2.64 → 0.05), and
   the metrics discriminate.
2. The **safety gate works and bites**: the CNN's 0.80 EMERGENCY recall (it
   missed an emergency) **fails** the 1.0 floor though its overall F1 passes —
   exactly the failure mode to catch *before* deployment.
3. CNN and rules produce **identical** (templated) narratives → in smoke, a CNN
   adds no narrative value over rules. The narrative question can only be
   answered by `distilled_qwen` vs `llm_teacher` in the real run.

## 7. Feasibility — CNN path

- **Deps:** none new (timm/onnx already present).
- **Train cost:** tiny head; ~minutes–1h for full epochs on one A6000 over
  RFMiD (image → triage). CPU works for the head but is slow at 224².
- **Artifact:** ~5–6 MB FP32, ~1.5 MB INT8 ONNX — trivially within the image.
- **Latency:** ~10 ms CPU at 64²; expect tens of ms at 224².
- **Upside:** smallest, fastest, fully offline, no new supply-chain surface.
- **Risk / open question:** (a) **EMERGENCY recall must hit 1.0** — the smoke run
  shows this is the hard part; mitigations = emergency-recall-weighted loss,
  threshold tuning, or a deterministic emergency override on top of the CNN.
  (b) Against a strong rule baseline the CNN's *marginal* triage value may be
  small — the real LLM teacher tells us whether there's nuance worth learning.
  (c) **No narrative** — text stays templated.

## 8. Feasibility — DistilledQwen path

- **Deps (new):** `transformers`, `accelerate` (+ `datasets`, `peft` for LoRA
  SFT; optional `bitsandbytes`/`auto-gptq` for INT4). A real supply-chain
  addition to review.
- **Base model:** Qwen2.5-0.5B-Instruct (smallest viable) or 1.5B (better
  narrative). 0.5B ≈ ~1 GB FP16 / ~0.4 GB INT4; 1.5B ≈ ~3 GB / ~1 GB INT4 —
  **10–100× the CNN**, the main deployment cost.
- **Recipe:** generate teacher traces (§5) → SFT/distill the small model on
  {case → JSON triage + narrative} → quantize (INT4/AWQ) → evaluate.
- **Compute:** trace gen = ~3.2k LLM call-pairs (API budget *or* hours on the
  self-hosted Qwen); SFT of a 0.5–1.5B model = a few GPU-hours on one A6000.
- **Latency:** generative; tens of tokens → **hundreds of ms–seconds on CPU**,
  fast on GPU. May not meet the 1.8 s edge budget on CPU without INT4 + short
  outputs.
- **Upside:** the **only** offline candidate that reproduces the narrative.
- **Risk:** size/latency for true edge; hallucinated diseases (grounding gate);
  trace cost; new deps.

### Side-by-side

| Dimension | CNN | DistilledQwen |
|---|---|---|
| New deps | none | `transformers`, `accelerate`, (+quant) |
| Artifact size | ~1.5–6 MB | ~0.4–3 GB |
| CPU p95 | ~tens ms | ~0.5–3 s |
| Triage | ✅ (recall risk) | ✅ |
| Narrative | ❌ template only | ✅ generative |
| Trace dataset | needed | needed |
| Fits mid-range Android | ✅ | ⚠️ INT4 + small only |

## 9. Go/No-Go — resolved (the real run was executed)

All four blockers below were cleared and the real comparison in §0 was run. Kept
for the record:

1. **Teacher source** — ✅ used the **self-hosted `Qwen3-8B-AWQ`** vLLM endpoint
   (`localhost:8011`); no external API key or cost. 80 traces cached to
   `traces.jsonl` (resumable).
2. **DistilledQwen base model + deps** — ✅ `transformers`/`peft`/`accelerate`
   present under system python; LoRA-distilled **Qwen2.5-0.5B-Instruct**
   (`outputs/distilled_qwen/`, 958 MB on disk / 1976 MB resident).
3. **GPU runtime** — ✅ solved. The project `.venv` torch (`2.12.0+cu130`) still
   reports `cuda.is_available() = False` (driver older than that build), so the
   real run used **system python (`/usr/bin/python3`, torch 2.6+cu124)** which
   sees the A6000s. Triton JIT needed `Python.h` (no sudo) — supplied via
   `apt-get download` + `dpkg -x` and a `CPATH` pointing at both header roots.
4. **Metric/gate sign-off** — ✅ §3 gates applied as-is. Caveat: EMERGENCY recall
   is **vacuous** on this sample (no EMERGENCY cases — see §0.5 and §10).

Outcome: see **§0**. **No Docker image is built until these results are signed
off.**

## 10. Environment caveats

- GPU works only under **system python**, not the project `.venv` (driver/torch
  mismatch, §9.3). The real run is reproducible via that interpreter.
- Real results are a **single held-out split, n=24** — directional, not a
  statistically tight estimate. A k-fold / larger-n pass would firm up the CNN's
  0.520 and the rule/teacher 1.000.
- The current classifier **over-detects** (threshold 0.53 chosen to get a varied
  triage mix), which skews the spread toward URGENT and yields **no EMERGENCY**
  cases in RFMiD-24 — so the safety gate is untested on real positives.
- Smoke synthetic images (§6) are probability encodings, **not** fundus photos —
  a learnability fixture, not a quality benchmark. §0 uses **real** RFMiD images.

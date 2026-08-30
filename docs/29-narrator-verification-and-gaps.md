# Verification of the §0.9 narrator sweep — and the gaps that remain

> Status: **review of [`28-reasoner-cnn-vs-distilledqwen.md`](28-reasoner-cnn-vs-distilledqwen.md) §0.9.**
> Verifies what that section claims, corrects what it got wrong, and lists what
> current (2026) practice would require before any of it is deployable. No
> Docker image is built; §0.9's narrator recommendation is **downgraded from
> "strictly beats" to "candidate pending re-measurement"** on the evidence below.

## TL;DR

The §0.9 *arithmetic* is sound — every number in the doc matches its JSON
artifact, there is no train/test leakage, tests and lint are green. The
§0.9 *conclusions* are not safe, for three independent reasons:

1. **The generation-rate comparison was confounded by a 200-token cap.** The
   teacher's JSON targets run to 252 tokens (p50 197, p90 230); ~40% exceed the
   200-token generation budget every eval script used, so outputs were truncated
   mid-JSON. That penalises whichever model writes longer prose — it is not a
   model property.
2. **The latency axis cannot rank these models at all.** All three narrators
   measure ~11 tok/s *regardless of size* — 25–102× below their memory-bandwidth
   roofline. A batch-1 decode that ignores a 3.7× difference in weight volume is
   measuring Python/launch overhead and GPU contention, not the model.
3. **The quality metric is blind to the failure mode that matters clinically.**
   `grounding` only asks "was every disease *named* actually detected". A
   claim-level audit (§2.4) shows the §0.9-recommended `smollm2_360m` is the
   **only** candidate that fabricates hard severity language — *"potentially
   life-threatening"*, *"permanent vision loss"* — for medium-confidence chronic
   findings, phrases appearing in **0 of 80** teacher traces. Every such
   narrative scored `grounding = 1.000`.

The net: SmolLM2-360M's *size* advantage (724 MB vs 988 MB) is real and holds.
Its claimed advantage on *reliability* evaporates once the truncation bug is
fixed (both hit 1.000), and on *faithfulness* it is arguably the worst of the
three. The recommendation should not stand on the evidence as it was gathered.

---

## 1. What verified clean

| Check | Method | Result |
|---|---|---|
| Train/test leakage | Compare `distill_qwen_reasoner.load_rows` split (no sort) against evaluators' `load_test_cases` split (sorts by `scan_id`) | ✅ **0 overlap** — the trace file happens to be written in sorted order, so both cut the same 56/24 |
| Doc ↔ artifact consistency | Every §0.9 table cell vs `narrator_arch_sweep.json` / `server_narrator_report.json` | ✅ exact match, all 6 rows |
| Test suite | `pytest -m "not slow"` | ✅ 22 passed |
| Lint | `ruff check` on all touched files | ✅ clean |
| Backward compatibility | New `ReasonerOutput.narrative_generated` field vs existing positional construction (`generalizability_test.py:294` passes 3 positional args) | ✅ safe — field is last, defaulted |
| Repo hygiene | 1.5 GB of new checkpoints | ✅ covered by `.gitignore` (`outputs/`) |
| Artifact provenance | `outputs/distilled_smollm2_{135m,360m}/` sizes vs `config.json` param math | ✅ 269 MB / 724 MB match bf16 param counts exactly |

The two code fixes made during the sweep are also confirmed correct and are
genuine bug fixes, independent of the conclusions:

- `_extract_json` raised `AttributeError` when a degraded model emitted valid
  JSON that parsed to a *bare string* rather than an object. Now raises
  `JSONDecodeError`, so the caller's rule-seed fallback engages instead of
  crashing the run. (This crash actually killed the first sweep.)
- `ReasonerOutput.narrative_generated` distinguishes model-authored text from
  the template fallback. Without it, `grounding` is unfalsifiable: the template
  is grounded *by construction*, so a narrator failing 100% of the time still
  scores 1.000. This is how `smollm2_135m_nf4`'s total failure hid in plain
  sight.

## 2. What verified wrong

### 2.1 The "like-for-like" claim was asserted, not measured — and was false as run

§0.9 states Qwen0.5B's generation rate "is also 1.00, i.e. it never failed to
parse in the original run either — so this is a like-for-like comparison". That
was inferred from the *absence* of a warning line in a **different run** of a
**different script**, not measured with the same instrumentation.

Measured directly on the same 24 cases with the same code path:

| Qwen0.5B bf16 | generation rate |
|---|---:|
| at the 200-token budget the sweep actually used | **0.833** |
| at a 384-token budget (clears the longest target) | **1.000** |

So the doc's number is right *in principle* and wrong *as measured*: under the
harness's own settings Qwen was the **less** reliable emitter, and the SmolLM2
comparison it was used to justify was not like-for-like. The claim only becomes
true after fixing the truncation bug in §2.2 — which is exactly why it should
have been measured rather than inferred.

### 2.2 The 200-token cap truncated ~40% of targets

```
teacher JSON target length: p50 197 tok, p90 230, max 252
targets exceeding the 200-token eval cap: 32/80 (40%)
```

Direct probe on `smollm2_135m_nf4` (6 held-out cases, budget 200 vs 512):

| budget | hit cap | parsed OK |
|---|---:|---:|
| 200 | 3/6 | 1/6 |
| 512 | 0/6 | **0/6** |

Two distinct conclusions, and they matter separately:

- **Cross-model generation-rate comparisons at 200 tokens are invalid** — half
  the probe cases were cut off by the budget, so the metric partly measures
  verbosity.
- **`smollm2_135m_nf4`'s failure is nonetheless real.** Given 512 tokens and no
  truncation at all, it still parsed 0/6. NF4 genuinely destroys this model's
  structured-output ability; that specific §0.9 finding survives.

All eval scripts now default to **384 tokens**. Re-measured, all six variants
through one code path:

| variant | size MB | grounding | gen rate (95% CI) | vs 200-tok |
|---|---:|---:|---:|---|
| `qwen0.5b_bf16` | 988.1 | 1.000 | 1.000 [0.86, 1.00] | 0.833 → **1.000** |
| `qwen0.5b_nf4` | 451.3 | 1.000 | 1.000 [0.86, 1.00] | — |
| `smollm2_360m_bf16` | 723.6 | 1.000 | 1.000 [0.86, 1.00] | unchanged |
| `smollm2_360m_nf4` | 251.8 | 0.972 | 0.958 [0.80, 0.99] | unchanged |
| `smollm2_135m_bf16` | 269.0 | 1.000 | 0.917 [0.74, 0.98] | unchanged |
| `smollm2_135m_nf4` | 109.8 | 1.000 | **0.000** [0.00, 0.14] | unchanged |

Only Qwen0.5B was truncation-limited; the other deficits are genuine. **Every
generation-rate CI overlaps every other except the 135M-NF4 collapse** — at
n=24 the sweep can only really distinguish "works" from "broken".

### 2.3 Latency numbers cannot rank the models

Batch-1 autoregressive decode is memory-bandwidth-bound: each token streams the
full weight set from HBM. On an RTX A6000 (768 GB/s):

| model | bf16 MB | roofline tok/s | @40% eff | **measured** | gap |
|---|---:|---:|---:|---:|---:|
| qwen0.5b | 988 | 777 | 311 | 12.3 | 25× |
| smollm2_360m | 724 | 1061 | 425 | 10.9 | 39× |
| smollm2_135m | 269 | 2855 | 1142 | 11.2 | **102×** |

The models differ by 3.7× in weight volume and measure **within 13% of each
other**. That is the signature of a fixed per-token overhead — the HF `generate`
Python loop, `attn_implementation="eager"`, no CUDA graphs, no `torch.compile`,
on a GPU running 100% co-tenant load — completely swamping the model. §0.7 half-
noticed this ("fp32 measured marginally faster… within contention noise") but
still drew size/latency conclusions from it. Every p50/p95 figure in §0.6, §0.7
and §0.9 should be treated as **non-comparative**.

### 2.4 `grounding` is blind to invented clinical severity

`narrative_metrics` iterates the known disease-code vocabulary and scores
mention→detected. Nothing else in the sentence is checked. Demonstrated on the
recommended model's own output:

> **teacher** — "These findings indicate multiple significant retinal pathologies
> with medium confidence levels."
>
> **smollm2_360m** — "These findings suggest a high risk of vision loss… indicates
> a complex and **potentially life-threatening** condition requiring prompt
> attention."

Findings were Optic Disc Pallor 60%, Laser Scars 58%, Diabetic Retinopathy 56%,
Central Serous Retinopathy 56% — chronic, non-acute, medium-confidence. The
phrase "life-threatening" appears in **0/80** teacher traces. `grounding` for
this case: **1.000**, because every disease *name* was real.

A claim-level audit (`scripts/audit_narrative_faithfulness.py`, bf16, 384-token
budget, same 24 cases) scores acuity divergence, quoted-probability fidelity and
omissions per case with Wilson CIs → `outputs/reasoner_comparison_real/faithfulness_audit.json`:

| narrator | gen rate | acuity divergence (95% CI) | terms it invented | bad probs | omissions |
|---|---:|---:|---|---:|---:|
| `qwen0.5b` | 1.000 | 0.500 [0.31, 0.69] | `immediate`, `rapid` | 0.0% | 0.0% |
| `smollm2_360m` | 1.000 | 0.208 [0.09, 0.41] | `immediate`, **`life-threatening`**, **`permanent vision loss`** | 0.0% | 0.0% |
| `smollm2_135m` | 0.917 | 0.182 [0.07, 0.39] | `immediate` | 0.0% | **13.6%** [0.05, 0.33] |

**This inverts the §0.9 recommendation on the axis that matters clinically.**
Ranked by how *often* a narrator departs from the teacher's acuity language,
`smollm2_360m` looks good (20.8% vs Qwen's 50%). Ranked by *what* it invents,
it is the worst of the three — and the only one that fabricates hard severity
claims: **"life-threatening"** and **"permanent vision loss"**, neither of which
appears anywhere in the 80 teacher traces. Qwen's divergences are frequent but
mild (`immediate`, `rapid`). `smollm2_135m` invents least but drops findings the
teacher reported in **13.6%** of cases — the opposite failure, under-reporting.

No candidate is clean, and the three fail in different directions. Every one of
these narratives scored `grounding = 1.000`.

> *Caveat on the metric:* the severity lexicon is deliberately broad
> ("severe", "acute", "rapid"…), so the *rate* is a screening signal for review,
> not a hallucination count — some hits are benign paraphrase. The **hard terms**
> (life-threatening, permanent vision loss, blindness, irreversible) are the ones
> to treat as defects, which is why the "terms invented" column matters more than
> the rate column. Both are proxies pending the NLI/LLM-judge cascade of §3.4.

### 2.5 Pre-existing doc inconsistency (inherited, not introduced)

§0's "**Decision:** Narrative → ship the grounded template now… The Docker image
needs **no LLM**" directly contradicts §0.7's decision to server-host a bf16
narrator, and now §0.9. §0's decision predates the edge-gate removal and was
never updated. It reads as the doc's conclusion because it sits in §0.

---

## 3. Gaps vs current practice

Ordered by leverage. (1)–(3) are the ones that would change the answer.

### 3.1 No constrained decoding — the whole failure mode is structurally avoidable

Every "generation rate" defect in §0.9 is a malformed-JSON defect. Grammar-
constrained decoding makes that **impossible by construction**: the schema is
compiled to a pushdown automaton and invalid tokens are masked at each step.
This is standard in vLLM and SGLang via [XGrammar](https://arxiv.org/pdf/2411.15100)
(<40 µs/token for JSON Schema; up to 3× faster than prior backends), Outlines,
or lm-format-enforcer. Reported results show a 1B model reaching 96.2% schema
accuracy with SFT + constrained decoding — the small-model + constraints
combination is precisely this use case.

Implication: `smollm2_135m_nf4`'s 0% is a *decoding-strategy* artifact as much as
a quantization one. The 110 MB variant may well be viable under constrained
decoding — and it is the closest thing to an edge-deployable narrator produced so
far. **This should be tested before any model is eliminated on reliability.**

### 3.2 bitsandbytes NF4 is a development convenience, not a serving quantizer

§0.6/§0.9 generalise "4-bit is not free" from bnb-NF4 alone. The re-measurement
shows the generalisation fails in *both* directions — NF4's damage tracks model
capacity, not the bit width:

| base model | params | NF4 gen rate | NF4 grounding |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 494M | **1.000** | 1.000 |
| SmolLM2-360M | 362M | 0.958 | 0.972 |
| SmolLM2-135M | 135M | **0.000** | — (all fallback) |

Smaller models carry less parameter redundancy, so identical quantization error
does disproportionate damage. NF4 costs the 0.5B model **nothing**.

That surfaces an option the §0.9 framing ("which base model at bf16?") hid
entirely: **`qwen0.5b_nf4` is 451 MB at grounding 1.000 and generation 1.000 —
smaller than `smollm2_360m_bf16` (724 MB) at equal reliability.** Quantizing the
incumbent beats swapping base models on the size axis. Its faithfulness is
unaudited (§2.4 covered bf16 only) and should be measured before it is adopted.

Separately, bnb NF4 remains the wrong *serving* choice regardless: it dequantises
on the fly into a bf16 matmul, so batch-1 decode pays per-token dequant overhead
— the consistent ~2× slowdown here is a property of that implementation.
AWQ+Marlin and GPTQ+ExLlamaV2 are well ahead for serving (same weights on the
right kernel can be ~10× the throughput). And **this project's own teacher is
`Qwen3-8B-AWQ`** — 4-bit, served under vLLM, working fine. AWQ/GPTQ narrator
variants are untested and should be measured before 4-bit is characterised at all.

### 3.3 No real serving stack

Everything was measured through `transformers.generate()` at batch 1 with eager
attention on a contended GPU. A server-hosted narrator would run under vLLM/
SGLang with CUDA graphs, paged KV cache and continuous batching. Until that is
measured, there is no defensible latency number, no throughput/cost figure, and
no basis for capacity planning. Report **TTFT and TPOT**, not a wall-clock p95
that includes model load and Python overhead.

### 3.4 Evaluation is far below clinical-generation standards

Current practice for clinical text scores **per-claim**, not per-answer: decompose
the output into atomic claims, check each against source, and run a cascade
(cheap heuristic → NLI entailment classifier → LLM judge on borderline only),
with human review kept in the loop. Published clinical-safety frameworks score
hallucination, factual consistency, completeness and coherence as separate
dimensions.

Current harness: one lexical disease-mention check. Missing — quoted-probability
fidelity, omission of reported findings, acuity/severity faithfulness, triage↔
narrative consistency, and any human review. `scripts/audit_narrative_faithfulness.py`
is a first step, not a substitute.

### 3.5 Statistical power

n=24 held-out, 56 training rows. A 0.917 vs 1.000 generation rate is a
**two-case** difference; the Wilson 95% intervals are **[0.742, 0.977]** and
**[0.862, 1.000]** — substantially overlapping. §0.9's "strictly beats" is not
supportable at this n.
CIs are now emitted by both scripts. RFMiD has ~3,200 images and the teacher is
self-hosted (no API cost), so the 80-trace dataset is a choice, not a constraint
— and the 135M model's reliability gap is at least as likely to be a data-volume
artifact as a capacity ceiling.

### 3.6 Base-model shortlist is a generation behind

SmolLM2 (2024) was chosen for vocabulary size. Worth benchmarking on the same
harness:

- **Gemma 3 270M** — explicitly positioned for task-specific fine-tuning, with
  *"unstructured text to structured output"* named as a target workload. 170M of
  its 270M params are embeddings (256k vocab), so ~540 MB bf16.
- **Qwen3-0.6B** — current-generation successor to the incumbent base model.
- **LFM2-350M**, **SmolLM3** — other current sub-1B options.

### 3.7 An untested path to the 60 MB edge gate

§0.6 concludes no generative model can meet 60 MB. That holds *as configured*,
but the binding constraint is the embedding table, and this task uses a tiny
token subset (disease names + clinical boilerplate). Vocabulary pruning is the
standard fix:

| model | vocab | embed params | prune to 4k → | bf16 | **4-bit** |
|---|---:|---:|---:|---:|---:|
| smollm2_135m | 49,152 | 28.3M (21%) | 108.6M | 217 MB | **~61 MB** |

That lands essentially *on* the 60 MB gate — and combined with §3.1's constrained
decoding (which fixes the reliability problem that disqualified the 4-bit 135M),
an edge-deployable generative narrator is plausible rather than impossible. This
is the single most valuable unexplored direction if edge deployment still matters.

### 3.8 Training recipe

Verified in `distill_qwen_reasoner.py`:

- **`--val-frac` splits data but validation is never scored** (`load_rows(..., "val")`
  is never called). No eval loss, no early stopping, no overfitting signal — on
  56 examples for 3 epochs.
- LoRA targets attention only (`q,k,v,o`); MLP modules (`gate/up/down`) are
  typically included where output *format and style* adherence is the objective.
- No seed → runs are not reproducible. No LR scheduler/warmup, no gradient
  clipping.
- Base models loaded without a pinned `revision=` — a supply-chain gap against
  the workspace's own dependency-pinning standard.
- Naming: this is **sequence-level KD** (SFT on teacher outputs), not logit
  distillation. That is the correct choice across different tokenizers, but the
  doc should say so.

*(Fixed during this review: `load_rows` now sorts by `scan_id`, so the training
split can no longer silently diverge from the evaluators' split if traces are
regenerated in a different order.)*

### 3.9 Regulatory — obligations are live *now*

EU AI Act high-risk obligations apply **from August 2026**, and AI systems for
diagnosis, clinical decision support and **patient triage** are classified
high-risk; MDR/IVDR devices needing Notified Body assessment get a transition to
August 2027. Transparency duties for AI-generated content also begin August 2026.

Concrete, verifiable gap in the artifacts:

| narrative source | carries AI-disclosure? |
|---|---|
| template fallback (0 MB) | ✅ "This is an AI-assisted screening result and requires specialist confirmation." |
| teacher traces (all 80) | ❌ **0/80** contain it (only 5/80 have any disclaimer-ish phrase) |
| every distilled student | ❌ inherits the teacher's omission |

**Replacing the template with any distilled narrator silently removes the
AI-disclosure and limitation statement from clinical output.** That is a
regression against Art. 50 transparency and Art. 14 human-oversight
expectations, and it is baked into the training data. Fix at the data layer
(regenerate traces with a mandated closing disclaimer) rather than by
post-processing.

Also unaddressed: Art. 12 logging/traceability of generated output, and Art. 15
accuracy/robustness evidence — for which §2.3's latency numbers and §3.5's n=24
would not currently suffice.

### 3.10 Nothing is integrated, and the fallback is silent in production

`grep` over `src/agents/` and `src/api/` finds **no reference** to the distilled
narrator or to `feat_logreg`. The live path is still Claude → Groq →
deterministic rules; `vllm_enabled` remains a config flag with no wiring. §0.7's
"final server stack" describes an evaluation result, not a deployed system —
there is no serving entry point, no config switch, and no integration test.

Relatedly: the template fallback that masked `smollm2_135m_nf4`'s total failure
is **only instrumented in the eval harness**. In the production path the same
degradation would be equally invisible. Per the workspace's observability
standard, narrative source (`generated` vs `template_fallback`) should be a
logged, alertable metric before any narrator ships.

---

## 3.11 Fixes applied — the compact narrator

Three of the gaps above compose into one change
(`scripts/build_compact_narrator.py`, `src/narrator/`), evaluated on the same 24
held-out cases → `outputs/reasoner_comparison_real/compact_narrator_report.json`.

**Prose-only output (closes §3.1 without a grammar library).** Every reliability
defect measured was a malformed-JSON defect, and the JSON existed only to carry
the triage fields — which `src/triage` now serves. A narrator that emits the
report as plain prose has no structured contract left to break. That is a
stronger guarantee than constrained decoding: a grammar prevents *invalid* JSON,
whereas having no JSON removes the failure mode. Measured generation rate
**1.000 at every precision**, including the 4-bit variant that previously scored
**0.000**.

**Vocabulary pruning (closes §3.7).** bitsandbytes leaves the embedding table
unquantized, which is the entire size gap — predicted and confirmed:

| | params | footprint |
|---|---:|---:|
| embedding, 49,152 × 576, bf16 (not quantized by bnb) | 28.3M | 56.6 MB |
| transformer body, 4-bit | ~106M | 53.1 MB |
| **predicted total** | | **110 MB** (measured 109.8) |

Pruning the vocabulary to the tokens this task can reach — **49,152 → 920** —
collapses the embedding to under 1 MB. Done by **id remapping, not tokenizer
surgery**: the original tokenizer is untouched and `keep_ids.json` maps between
its id space and the model's compact one, which is exact and reversible where
rewriting BPE merges is easy to get subtly wrong.

| variant | size | gen rate | omission | bad probs | acuity drift | words |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 213.5 MB | 1.000 | **0.000** | 0.000 | 0.250 | 68 |
| **nf4** | **54.2 MB** | 1.000 | **0.417** | 0.167 | 0.583 | 89 |

**The 60 MB edge gate is now reachable — but 4-bit should not ship on these
numbers.** 54.2 MB clears a gate §0.6 called impossible, and generation no longer
fails. The cost is content fidelity: the 4-bit model drops findings the teacher
reported in **42%** of cases and misquotes probabilities in 17%, while rambling
(89 words vs 68). That is a *worse* failure mode than the one it fixes — it now
always produces fluent text, so omissions are harder to notice.

**There is no 60-100 MB option with this stack.** The obvious middle point was
int8, and it misses on both axes:

| precision | size | one-case latency | verdict |
|---|---:|---:|---|
| bf16 | 213.5 MB | 1.4 s | quality-preserving, too large for the edge |
| int8 | **107.3 MB** | **37.5 s** | over the band *and* ~8x slower than 4-bit |
| nf4 | 54.2 MB | 4.8 s | under the band, fast, but fidelity degrades |

bitsandbytes' int8 path decomposes each matmul into mixed precision, which costs
far more at batch 1 than the memory it saves — 37.5 s/case made a 24-case
evaluation impractical to even complete. This is the same conclusion as §3.2 from
the other direction: **bnb is a development convenience, not a serving
quantizer.** A genuine 60-100 MB target needs AWQ/GPTQ (which quantize with fused
kernels) or GGUF `Q5_K_M`-class weights under llama.cpp, neither of which is
installed here — that is the next experiment, not a settled result.

**bf16's clean held-out scores do not generalise off-distribution.** The audit
set is dominated by multi-finding cases resembling training data. Given a
*single*-finding case by hand, the bf16 model produced:

> "a **75% detection** of diabetic retinopathy with a **prevalence of 88%**"

The input was one finding at 88% confidence. Both figures are wrong and
"prevalence" is invented — the exact `prob_infidelity` failure the audit scored
at 0.000 on the held-out split. One ad-hoc probe is not a measurement, but it is
enough to show the 0.000 is a property of that split rather than of the model,
and it is why `NARRATOR_ENABLED` defaults to **false**.

**A bug this work surfaced.** The vocabulary was first built from tokens observed
in the 240 traces — which covered 14 of 16 harness names, while the classifier
can emit **45 classes**. Any unseen disease name would have had its tokens
silently dropped, handing the model a corrupted prompt with no error. The
vocabulary is now seeded from the production `DISEASE_NAMES` map, unmappable
tokens fall back to byte-level encoding rather than being dropped, and the
evaluation asserts that **all 45 class names survive the round trip** (they do).

## 3.13 Format evaluation — AWQ, GPTQ and GGUF

§3.2 recommended AWQ/GPTQ over bitsandbytes. Evaluating that properly changes the
recommendation, for a reason specific to this model.

### The size problem is the embedding table, not the quantizer

A 135M SmolLM2 is **21% embedding** (49,152 × 576 = 28.3M params). What each
format does with that table decides the footprint:

| format | body | embedding | projected total |
|---|---|---|---:|
| bnb NF4 | 4-bit | **fp16, not quantized** | 110 MB (measured 109.8) |
| bnb int8 | int8 | **fp16, not quantized** | 107 MB (measured 107.3) |
| AWQ / GPTQ (W4A16) | 4-bit | **fp16, conventionally excluded** | ~110 MB |
| **GGUF `Q4_K_M`** | ~4.5 bit | **quantized too** | **~80 MB** |

**AWQ and GPTQ would not beat bitsandbytes on size here.** They exclude the
embedding from quantization exactly as bnb does, so they land at the same ~110 MB
— their advantage is *throughput* (fused Marlin/ExLlama kernels), not footprint.
For an embedding-dominated small model, that advantage does not address the
constraint.

**GGUF is the only one of the three that quantizes the embedding**, which makes
`Q4_K_M`/`Q5_K_M` the most promising untested route into the 60-100 MB band
*without* the vocabulary surgery of §3.11 — and therefore without the fidelity
loss that 4-bit-plus-pruning incurred. That is the experiment worth running next.

### Vocabulary pruning trades away format compatibility

The pruned checkpoint deliberately breaks the model↔tokenizer correspondence:

```
config.vocab_size (model output dim) : 920
tokenizer vocabulary entries         : 49152
```

`src/narrator/compact.py` bridges them with an explicit id remap. Any tool that
assumes `vocab_size == len(tokenizer)` will mishandle the checkpoint:

* **GGUF** bakes the tokenizer into the file, so conversion would either fail or
  emit a file whose token ids do not mean what the weights expect.
* **AWQ/GPTQ** quantize weights only and would run mechanically, but the artifact
  would still need the custom wrapper — and the *reason* to adopt them is serving
  under vLLM/SGLang/TGI, which assume standard tokenization. The cost would be
  paid and the benefit lost.

So pruning and standard-format quantization are **alternative** routes to a small
narrator, not complementary ones. Pruning wins on footprint (54 MB) and loses
ecosystem compatibility; GGUF keeps compatibility and should reach ~80 MB.

### Tooling status on this host (measured, not assumed)

| tool | status |
|---|---|
| `autoawq` 0.2.9 | installs, **cannot import**: `ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'` against transformers 4.57.1. Upstream is unmaintained; superseded by `llm-compressor`/`compressed-tensors`. |
| `compressed-tensors` 0.11.0 | present (vLLM dependency) — the current AWQ/GPTQ path |
| `gptqmodel` | not installable in the time budget here |
| `gguf` (writer lib) | present, but can only write F16 — K-quants need `llama-quantize` |
| `llama.cpp` binaries | absent (`llama-quantize`, `convert_hf_to_gguf.py`) |

None of the three could be *measured* on this host. The size projections above
are arithmetic from published quantization layouts, not measurements, and are
labelled as such.

The Space had been in `CONFIG_ERROR` since 23 June. Two independent bugs, both in
`.github/workflows/deploy-hf-spaces.yml`, both now fixed:

1. **`short_description` was 78 characters** against a 60-character Hub limit,
   and contained an unquoted colon (invalid YAML). `/api/validate-yaml` rejected
   the README on every deploy, so the Space never got as far as building.
2. **`rsync` ran without `--delete`**, so files from older deploys survived. Once
   (1) was fixed and a build actually started, it failed on a stale
   `frontend/src/app/page.tsx` that referenced a store field
   (`AppState.currentPage`) removed long ago. The current tree has no such file.

With both fixed the Space builds and serves: `/health` returns
`{"status":"healthy","model_loaded":true,"diseases_count":45}`.

## 4. Recommended sequence

1. ~~Finish the 384-token re-measurement~~ ✅ **done** — §2.2. "Strictly beats"
   withdrawn; `qwen0.5b_nf4` (451 MB) emerged as the smallest fully-reliable
   variant and needs a faithfulness audit before it can be compared properly.
2. ~~Add constrained decoding~~ ✅ **superseded by §3.11** — moving triage to the
   3 KB head let the narrator emit prose instead of JSON, which removes the
   parse-failure mode outright rather than constraining it. Generation rate is
   1.000 at every precision, so size and faithfulness are now the discriminators
   as predicted. Constrained decoding is still the right tool if a structured
   narrator is ever needed again.
3. **Re-measure latency under vLLM** with CUDA graphs; report TTFT/TPOT.
4. **Regenerate traces at scale** (hundreds–thousands, teacher is free) **with a
   mandated disclaimer**, fixing §3.5 and §3.9 together.
5. **Then** re-run the base-model sweep including Gemma 3 270M and Qwen3-0.6B,
   with AWQ/GPTQ rather than bnb-NF4.
6. Only after that, revisit the Docker image — and wire an integration path plus
   fallback-rate telemetry as part of it.

---

## Sources

- [XGrammar: Flexible and Efficient Structured Generation Engine for LLMs](https://arxiv.org/pdf/2411.15100)
- [Structured Decoding in vLLM: a gentle introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [Structured Outputs — vLLM docs](https://docs.vllm.ai/en/v0.8.4/features/structured_outputs.html)
- [JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models](https://arxiv.org/pdf/2501.10868)
- [GPTQ vs AWQ vs GGUF: Which 4-Bit to Pick in 2026](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs)
- [LLM Quantization Guide: GGUF vs AWQ vs GPTQ vs bitsandbytes Compared (2026)](https://www.premai.io/blog/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)
- [A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation (npj Digital Medicine)](https://www.nature.com/articles/s41746-025-01670-7)
- [Automating Evaluation of AI Text Generation in Healthcare with LLM-as-a-Judge (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12045442/)
- [AI hallucination evaluations: metrics and methods that work in 2026 (Braintrust)](https://www.braintrust.dev/articles/ai-hallucination-evaluations-metrics-methods-2026)
- [Introducing Gemma 3 270M: the compact model for hyper-efficient AI](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [The Best Open-Source Small Language Models (SLMs) in 2026 (BentoML)](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [EU AI Act explained: what healthcare organisations need to know](https://tandemhealth.ai/resources/knowledge/eu-ai-act-explained-what-healthcare-organisations-need-to-know)
- [The AI Act Omnibus: what the 2026 EU rules mean for medical device and IVD manufacturers](https://patientguard.com/the-ai-act-omnibus-explained-what-the-2026-eu-rules-mean-for-medical-device-and-ivd-manufacturers/)
- [FDA: Artificial Intelligence in Software as a Medical Device](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)

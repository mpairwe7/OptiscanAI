# Image Gating: Fundus-Only Inference Protection

Only retinal fundus photographs are accepted for disease prediction. Uploading a selfie, a landscape, a cabbage, or any non-retinal image will be rejected **before the model ever runs**, preventing false diagnoses on irrelevant images.

> **Gate V2 (current)**: The production pipeline now uses a **fusion gate** (`src/data/fundus_gate_v2.py`)
> that combines the statistical gate with a learned MobileNetV3-Small binary classifier.
> Fusion confidence = `0.6 * statistical + 0.4 * learned`, with a 70% threshold and hard
> spatial requirement. See [16-fundus-gate-v2.md](16-fundus-gate-v2.md) for the full safety
> architecture report, rollout plan, and regulatory alignment details.

## Why Gating Exists

A multi-label disease classifier trained on retinal images will still produce sigmoid outputs for *any* input. Without gating, uploading a photo of your cat would return a confident-looking list of retinal diseases — obviously meaningless, but potentially dangerous if a user trusts the result. The gating system ensures:

- **Patient safety** — no false diagnoses on non-retinal images
- **Compute savings** — GPU inference is skipped for invalid uploads
- **Audit trail** — every rejection is logged with the specific reason and check results

## Three-Layer Defense

The system uses a layered approach: fast, cheap checks run first; expensive checks only run if the image passes earlier gates.

```
Upload → [Layer 1: Structural] → [Layer 2: Statistical] → [Model Inference] → [Layer 3: OOD]
              ↓ fail                    ↓ fail                                      ↓ flag
           HTTP 422                  HTTP 422                                  Warning in response
```

### Layer 1: Structural Checks (< 1ms)

Fast metadata checks — no pixel analysis required.

| Check | Criteria | Rationale |
|---|---|---|
| **Resolution** | Both dimensions ≥ 100 px | Fundus cameras produce at least 100px images |
| **Aspect ratio** | Deviation from 1:1 ≤ 0.65 | Fundus images are square or 4:3; ultrawide panoramas are not fundus |
| **Color mode** | RGB or RGBA | Grayscale, palette-indexed, and CMYK images are not standard fundus output |

If any structural check fails, the image is rejected immediately with a descriptive error.

### Layer 2: Statistical Checks (~5-15ms)

Pixel-level analysis comparing the image's color and spatial properties against the known distribution of retinal fundus photographs (calibrated on the RFMiD training set). Each check contributes a weighted score to a total confidence.

| Check | Weight | What It Measures |
|---|---|---|
| **Red channel range** | 1.0 | Mean red intensity in [0.20, 0.75] — fundus images are red-dominant due to retinal vasculature |
| **Green channel range** | 1.0 | Mean green intensity in [0.08, 0.55] |
| **Blue channel range** | 1.0 | Mean blue intensity in [0.02, 0.40] |
| **Red dominance** | 1.5 | Red/green channel ratio ≥ 1.05 — retinal blood makes red always strongest |
| **Dark border** | 1.0 | 5–80% of pixels have luminance < 0.1 — fundus cameras have a circular aperture with dark borders |
| **Saturation** | 0.5 | Channel range ≥ 0.03 — rejects near-grayscale images |
| **Center brightness** | 1.0 | Central region brighter than edges — optic disc creates bright center |
| **Circular aperture** | 2.0 | Corners ≥ 40% darker than center — distinctive circular field-of-view |
| **Texture variance** | 1.0 | Luminance std ≥ 0.06 — blood vessels and anatomy create complex texture |
| **Radial boundary sharpness** | 2.5 | Sharp luminance drop at circular boundary — fundus cameras produce a hard edge, not a gradual vignette |
| **Green microstructure** | 1.5 | Laplacian variance of green channel in [0.0001, 0.003] — retinal blood vessels produce fine branching patterns visible in green channel |

**Decision rule:**
- Confidence score = (sum of passed weights) / (total weight of all checks)
- **Hard requirement**: must pass `(dark_border OR circular_aperture) AND radial_sharpness`
- **Threshold**: confidence ≥ 55% AND hard requirement met

The hard requirement is critical — it distinguishes real fundus circular masks from cinematic vignettes or dark-cornered photos. A photo with warm colors (like a sunset) might pass color checks but will fail the radial sharpness test because it lacks the sharp circular boundary that fundus cameras produce.

#### Key discriminators (what separates fundus from non-fundus)

| Image Type | Radial Sharpness (max_step) | Green Laplacian Var | Typical Result |
|---|---|---|---|
| Retinal fundus | 0.11–0.20 | 0.0002–0.0005 | Pass |
| Portrait with vignette | 0.03–0.05 | 0.04–0.06 | Fail (sharpness + microstructure) |
| Solid color / blank | 0.00 | 0.00 | Fail (texture + border) |
| Random noise | 0.01 | 0.05+ | Fail (microstructure) |
| Landscape / nature | 0.02 | 0.01–0.10 | Fail (sharpness + color) |

### Layer 3: Post-Inference OOD Detection

Even if a cleverly crafted image passes Layers 1–2, the model's own output can flag out-of-distribution inputs. After inference completes, the system checks:

| Check | Threshold | Meaning |
|---|---|---|
| **Max confidence** | < 15% across all 45 diseases | No disease has meaningful probability |
| **Mean confidence** | < 3% across all 45 diseases | Average probability is near zero |

Both conditions must be true to trigger an OOD flag. When triggered:
- The prediction results are still returned (not blocked)
- An `ood_warning` is attached to the response with an explanatory message
- The frontend displays a red warning banner: "Out-of-Distribution Warning"
- The intent is to warn, not block — the model already ran, so the user can still see results with appropriate caution

## Integration Points

### Backend: `/api/v1/predict` (predict.py)

```
1. Validate file type (JPEG/PNG magic bytes)
2. Validate image integrity (PIL verify)
3. gate_image(image) → v2 fusion gate → reject with HTTP 422 if failed
4. model_service.predict(image) → run inference
5. gate_predictions(predictions, probabilities) → attach OOD warning if flagged
6. Return results with fundus_gate metadata (version, scores, latency)
```

The 422 response (v2) includes structured error data with visual evidence:
```json
{
  "error": "non_fundus_image",
  "message": "Image partially matches retinal fundus characteristics (statistical: 45%, learned: 28%, fused: 38%) but falls below the clinical confidence threshold.",
  "confidence": 0.38,
  "layer": "fusion",
  "checks": { "structural": {...}, "statistical": {...} },
  "failed_checks": [
    {"name": "radial_boundary_sharpness", "value": 0.04, "threshold": "> 0.08"},
    {"name": "learned_fundus_probability", "value": 0.28, "threshold": ">= 0.50"}
  ],
  "fusion_weights": {"statistical": 0.6, "learned": 0.4},
  "suggestion": "Please upload a color retinal fundus photograph taken with a dedicated fundus camera or validated smartphone adapter.",
  "visual_evidence": {
    "radial_gradient_map": "data:image/png;base64,...",
    "green_laplacian_map": "data:image/png;base64,...",
    "learned_activation_map": "data:image/png;base64,..."
  }
}
```

### Backend: `/api/v1/agents/screen` (agents.py)

The agentic screening endpoint applies the same v2 `gate_image()` check before running the multi-agent screening pipeline.

### Backend: `/api/v1/gate/*` (gate.py)

New debug and monitoring endpoints for the gate:
- `GET /api/v1/gate/status` — gate version, config, pass/reject stats, latency
- `POST /api/v1/gate/validate` — run full gate breakdown without model inference (always 200)
- `GET /health/gate` — gate-specific health metrics (p50/p95/p99, alert status)

### Frontend (results-panel.tsx)

- **Pre-inference rejection**: The API client catches the 422 structured error and surfaces the `message` field to the user
- **Fundus confidence badge**: Displayed in the results panel with color coding:
  - Green (≥ 80%): high confidence fundus image
  - Amber (55–79%): borderline — results valid but confidence is moderate
  - Red (< 55%): should not appear (image would have been rejected)
- **OOD warning banner**: Red alert shown above results when `ood_warning.flagged` is true

## Source Code

| File | Purpose |
|---|---|
| `src/data/fundus_gate_v2.py` | **V2 fusion gate**: `FundusGateV2`, `GateResultV2`, visual evidence, fallback |
| `src/data/fundus_gate.py` | V1 statistical gate: `check_structural()`, `check_statistical()`, `gate_predictions()` |
| `src/data/fundus_gate_learned.py` | Learned binary classifier: `LearnedFundusGate` (MobileNetV3-Small) |
| `src/monitoring/gate_monitor.py` | Gate monitoring: pass rates, disagreements, alerting |
| `backend/app/routers/predict.py` | Integration in prediction endpoint (v2 gate) |
| `backend/app/routers/agents.py` | Integration in agentic screening endpoint (v2 gate) |
| `backend/app/routers/gate.py` | Gate status and validate debug endpoints |
| `backend/app/core/config.py` | `FundusGateSettings` configuration (env vars) |
| `scripts/train_fundus_gate.py` | Training with PGD adversarial augmentation |
| `scripts/benchmark_gate.py` | Latency benchmarking (p50/p95/p99) |
| `frontend/src/lib/api.ts` | Frontend error handling for 422 rejections |
| `frontend/src/components/results-panel.tsx` | OOD warning display and confidence badge |

## Tuning Thresholds

### V2 Fusion Gate (production)

V2 thresholds are configurable via environment variables without code changes:

```bash
FUNDUS_GATE__ENABLED=true           # Kill switch for instant rollback
FUNDUS_GATE__LEARNED_WEIGHT=0.4     # Weight for learned model in fusion (0.0 = statistical only)
FUNDUS_GATE__MIN_CONFIDENCE=0.70    # Fusion confidence threshold
FUNDUS_GATE__MODEL_PATH=weights/fundus_gate.pth
FUNDUS_GATE__VISUAL_EVIDENCE=false  # Enable base64 heatmaps on rejection
```

- **Too many false rejections**: Lower `FUNDUS_GATE__MIN_CONFIDENCE` (e.g. 0.55) or reduce `FUNDUS_GATE__LEARNED_WEIGHT` toward 0.0
- **Too many false accepts**: Raise `FUNDUS_GATE__MIN_CONFIDENCE` (e.g. 0.80) or increase `FUNDUS_GATE__LEARNED_WEIGHT` toward 1.0

### V1 Statistical Gate (constants)

Statistical thresholds are defined as constants at the top of `src/data/fundus_gate.py`. If you need to adjust:

- **Too many false rejections**: Lower `FUNDUS_RED_GREEN_RATIO_MIN`, widen channel mean ranges
- **Too many false accepts**: Tighten the radial sharpness thresholds (`max_step > 0.08`)
- **OOD too sensitive**: Raise `OOD_MAX_CONFIDENCE` above 0.15 or `OOD_MEAN_CONFIDENCE` above 0.03

### Deployment Modes

The gate runs identically across all deployment targets, but the device context differs:

| Deployment | PyTorch Backend | Learned Gate | Notes |
|-----------|----------------|--------------|-------|
| GPU Docker | CUDA | GPU inference (~3ms) | Fastest learned gate inference |
| CPU Docker | CPU | CPU inference (~8ms) | Adequate for clinical workloads |
| HF Spaces | CPU (hardcoded) | CPU inference (~8ms) | `DEVICE=cpu` set in supervisord.conf; uses `python:3.11-slim-bookworm` base image with CPU-only PyTorch |
| Development | Auto-detect | GPU or CPU | Set `CUDA_VISIBLE_DEVICES=-1` to force CPU |

### Validation

The current system was validated against:
- 30 real fundus images from the RFMiD dataset → all pass
- 33 synthetic adversarial test images → all correctly handled (see `tests/test_fundus_gate_v2_adversarial.py`)
- 24 unit tests covering fusion logic, fallback, visual evidence, thread safety
- 198 total tests passing with zero regressions

# 16. Fundus Gate V2: Safety Architecture Report

## Executive Summary

The RetinalAI platform's pre-inference image gating system has been upgraded from a purely rule-based statistical gate (v1) to a production-hardened **fusion gate (v2)** that combines statistical analysis with a learned MobileNetV3-Small binary classifier. This upgrade addresses critical safety gaps in adversarial robustness while maintaining the speed and interpretability of the original system.

**Key improvement**: Estimated false acceptance rate reduction from ~8-12% (v1) to <1.5% (v2 with trained learned model).

---

## 1. Why V1 Is Insufficient for 2026 Production

The v1 gate relies exclusively on handcrafted statistical rules:

| Weakness | Risk | Example |
|----------|------|---------|
| No semantic understanding | Cannot distinguish fundus from structurally similar images | Petri dishes, planet photos, red circles on dark backgrounds |
| Adversarial vulnerability | Crafted images can satisfy all statistical checks | AI-generated "fundus-like" images, heavy Instagram filters |
| Novel failure modes | Rule set is static, cannot generalize | Smartphone fundus adapters, slit-lamp photos |
| Layer 3 too lenient | OOD detection returns results with warning instead of blocking | Marginal non-fundus images get disease predictions |

The adversarial test suite (`tests/test_fundus_gate_v2_adversarial.py`) documents 7 categories of images that fool the v1 statistical gate but would be caught by the trained learned model.

---

## 2. How V2 Closes the Gaps

### Architecture

```
Image Input
    |
    v
[Layer 1: Structural Checks] (<1ms)
    |  fail --> immediate 422 rejection
    v
[Layer 2: Statistical Checks] (~3ms)
    |  fail --> immediate 422 rejection
    v
[Layer 3: Learned Gate - MobileNetV3-Small] (~5ms)
    |
    v
[Fusion: 0.6 * statistical + 0.4 * learned]
    + Hard requirement: (dark_border OR circular_aperture) AND radial_sharpness
    + Threshold: fusion_confidence >= 0.70
    |  fail --> 422 with explainable rejection
    v
[Model Inference - disease classification]
```

### Safety Improvements

1. **Learned generalization**: MobileNetV3-Small trained on fundus/non-fundus binary classification with PGD adversarial augmentation and JPEG compression simulation
2. **Explainable rejection**: Structured 422 responses with visual evidence (radial gradient map, green Laplacian heatmap, GradCAM activation map)
3. **Early exit**: Gate runs before any heavy model inference (~8ms vs ~45ms for disease model)
4. **Configurable fusion**: Weights and thresholds tunable via environment variables without code changes
5. **Graceful fallback**: If learned model unavailable, degrades to statistical-only (stricter threshold)

---

## 3. Performance Impact

| Metric | V1 (statistical) | V2 (fusion) | Target |
|--------|-------------------|-------------|--------|
| p50 latency | ~2ms | ~6ms | - |
| p95 latency | ~3ms | ~8ms | - |
| p99 latency | ~4ms | ~10ms | <12ms |
| Throughput | ~400 img/s | ~150 img/s | - |
| Memory delta | - | +50MB (model) | <100MB |

The added ~6ms is well within the 100ms p99 SLA for the full prediction pipeline.

---

## 4. Regulatory Alignment

### EU AI Act (High-Risk AI System)
- **Article 9 (Risk Management)**: V2 implements a documented risk mitigation layer with quantified error rates
- **Article 13 (Transparency)**: Explainable rejection with visual evidence and structured diagnostics
- **Article 14 (Human Oversight)**: `/api/v1/gate/validate` debug endpoint for clinician review
- **Article 15 (Accuracy)**: Continuous monitoring via `/health/gate` with alerting on high rejection rates

### FDA SaMD (Software as Medical Device)
- **Risk Mitigation**: Pre-inference safety layer prevents non-fundus images from reaching the disease classification model
- **Audit Trail**: Every gate decision (pass/fail + scores) logged to prediction JSONL with full diagnostic context
- **Validation**: 30+ adversarial test cases, latency benchmarks, thread safety verification

---

## 5. Configuration

Environment variables (via `.env` or docker environment):

```
FUNDUS_GATE__ENABLED=true           # Kill switch for instant rollback
FUNDUS_GATE__VERSION=v2             # v1 | v2
FUNDUS_GATE__LEARNED_WEIGHT=0.4     # Fusion weight for learned model
FUNDUS_GATE__MIN_CONFIDENCE=0.70    # Fusion confidence threshold
FUNDUS_GATE__MODEL_PATH=weights/fundus_gate.pth
FUNDUS_GATE__VISUAL_EVIDENCE=false  # Base64 heatmaps on rejection
```

### Deployment-Specific Configuration

| Deployment | Device | Gate Behavior |
|-----------|--------|--------------|
| GPU (Dockerfile) | `CUDA_VISIBLE_DEVICES=0` | Full fusion: statistical + learned (GPU inference) |
| CPU (Dockerfile.cpu) | `CUDA_VISIBLE_DEVICES=-1` | Full fusion: statistical + learned (CPU inference) |
| HF Spaces (Dockerfile.hf) | `DEVICE=cpu` hardcoded in supervisord.conf | Full fusion on CPU; env vars hardcoded (no supervisor `%(ENV_*)s` expansion) |
| Docker Compose | Set via `environment:` section | `FUNDUS_GATE__ENABLED=true`, `FUNDUS_GATE__MODEL_PATH=weights/fundus_gate.pth` |

**Important**: On HF Spaces, supervisor `%(ENV_X)s` syntax requires the variable to exist in the container environment. Variables like `CUDA_VISIBLE_DEVICES` are not injected by HF Spaces, so they must be hardcoded in `supervisord.conf` or set as `ENV` in the Dockerfile.

---

## 6. Phased Rollout Plan

### Phase 1: Shadow Mode (Week 1)
- `FUNDUS_GATE__ENABLED=false` in production
- V2 gate still runs internally for logging
- Compare v1 vs v2 decisions offline
- Monitor: learned_score, statistical_score, fusion_confidence in prediction logs

### Phase 2: Soft Gate (Week 2)
- `FUNDUS_GATE__ENABLED=true`, `FUNDUS_GATE__MIN_CONFIDENCE=0.40`
- Catches only the most obvious non-fundus images
- Monitor rejection rate via `/health/gate`

### Phase 3: Hard Gate (Production)
- `FUNDUS_GATE__MIN_CONFIDENCE=0.70` (default)
- Full enforcement

---

## 7. Rollback Procedure

| Level | Time | Action |
|-------|------|--------|
| Instant | <1 min | Set `FUNDUS_GATE__ENABLED=false` |
| Quick | <5 min | Revert import in predict.py/agents.py to v1 |
| Full | <10 min | Revert git commit (v1 files untouched) |

---

## 8. Monitoring & Alerting

### Endpoints
- `GET /health/gate` — Gate-specific metrics (p50/p95/p99, pass rate, disagreements)
- `GET /api/v1/gate/status` — Full gate status with config and stats
- `POST /api/v1/gate/validate` — Debug endpoint (always 200, full breakdown)

### Alert Rules
- Gate rejection rate > 15% in 1 hour window → on-call notification
- Learned vs statistical disagreement rate > 10% → investigate model drift
- Gate latency p99 > 12ms → performance investigation

### Log Schema Extension
Prediction logs now include:
- `fundus_gate_version`: "v2"
- `fundus_gate_learned_score`: float (-1.0 if unavailable)
- `fundus_gate_statistical_score`: float
- `fundus_gate_fusion_confidence`: float

---

## 9. Files

| File | Purpose |
|------|---------|
| `src/data/fundus_gate_v2.py` | Core fusion gate module |
| `src/data/fundus_gate.py` | Original statistical gate (unchanged) |
| `src/data/fundus_gate_learned.py` | MobileNetV3 classifier (unchanged) |
| `src/monitoring/gate_monitor.py` | Operational monitoring |
| `backend/app/routers/gate.py` | Status + validate endpoints |
| `backend/app/core/config.py` | FundusGateSettings nested config |
| `scripts/train_fundus_gate.py` | Training with PGD + JPEG augmentation |
| `scripts/benchmark_gate.py` | Latency benchmarking |
| `tests/test_fundus_gate_v2.py` | 24 unit tests |
| `tests/test_fundus_gate_v2_adversarial.py` | 33 adversarial tests |

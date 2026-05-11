# Backend Setup, Structure & Functionality

> Comprehensive documentation for the RetinalAI Clinical Screening Platform backend.
> Covers architecture, all API endpoints, core services, middleware, security, configuration, and deployment.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Core Services](#core-services)
6. [Middleware Stack](#middleware-stack)
7. [API Reference](#api-reference)
8. [Authentication & Authorization](#authentication--authorization)
9. [Security](#security)
10. [Configuration Reference](#configuration-reference)
11. [Logging & Audit Trail](#logging--audit-trail)
12. [Error Handling](#error-handling)
13. [Deployment](#deployment)
14. [Key Design Decisions](#key-design-decisions)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI Application (v3.0.0)                     │
│                    Lifespan: startup → model.load()                  │
│                              shutdown → model.unload()               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────── Middleware Stack ───────────────────────────┐ │
│  │  1. Security Headers (HSTS, X-Frame-Options, CSP, etc.)        │ │
│  │  2. Rate Limiter (token-bucket, 60 req/min per IP)             │ │
│  │  3. Request ID (UUID tracing + latency headers)                │ │
│  │  4. GZIP Compression (responses > 1 KB)                        │ │
│  │  5. CORS (configurable origins)                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────── 11 API Routers ────────────────────────────┐ │
│  │  health      ── /health, /health/model, /health/gate, /        │ │
│  │  predict     ── /api/v1/predict, /api/v1/diseases              │ │
│  │  predict_edge── /api/v1/predict/onnx,coreml,quantized (Ph 3)  │ │
│  │  auth        ── /api/v1/auth/token                             │ │
│  │  review      ── /api/v1/review/*                               │ │
│  │  clinical    ── /api/v1/clinical/*                             │ │
│  │  explain     ── /api/v1/explain/*                              │ │
│  │  analytics   ── /api/v1/system/info, /api/v1/analytics/summary│ │
│  │  agents      ── /api/v1/agents/*                               │ │
│  │  governance  ── /api/v1/governance/* (drift,AL,fairness,audit) │ │
│  │  gate        ── /api/v1/gate/status, /api/v1/gate/validate     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────── Core Services ─────────────────────────────┐ │
│  │  ModelService     ── singleton, model loading + inference       │ │
│  │  ModelExplainer   ── lazy-init, GradCAM/LIME/SHAP/IG/ELI5     │ │
│  │  PredictionLogger ── append-only JSONL audit trail             │ │
│  │  JWT Auth         ── HMAC-SHA256 token signing + RBAC          │ │
│  │  KnowledgeGraph   ── 45-disease clinical reasoning graph       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────── External Integrations ─────────────────────┐ │
│  │  Claude API (primary LLM) → Groq API (fallback) → rules       │ │
│  │  AgentOrchestrator (LangGraph-based multi-step screening)      │ │
│  │  Event Bus (SCAN_ANALYZED → agent reactions)                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
Client Request
  → Security Headers middleware
    → Rate Limiter middleware
      → Request ID middleware (assigns UUID, starts timer)
        → GZIP middleware
          → CORS middleware
            → Router handler
              → Auth dependency (if AUTH_ENABLED)
                → Business logic (model inference, KG reasoning, etc.)
                  → PredictionLogger (async audit write)
                    → Event Bus (notify agents)
                      → JSON Response (with X-Request-ID, X-Response-Time-Ms)
```

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Framework | FastAPI 0.115+ | Async REST API with lifespan management |
| Package Manager | UV (`pyproject.toml`) | Dependency resolution and virtual env |
| Model Serving | PyTorch 2.0+ | ViGNN inference with CUDA auto-detection |
| Config | Pydantic Settings 2.0+ | Type-safe env vars + `.env` file loading |
| Authentication | JWT (HMAC-SHA256) | Toggleable token-based auth with RBAC |
| Logging | Structured JSON | Machine-parseable logs for aggregation (ELK, CloudWatch) |
| Compression | GZIP | Automatic response compression for base64 payloads |
| Explainability | GradCAM, LIME, SHAP, Captum, ELI5 | 5 XAI methods with lazy initialization |
| Agentic AI | Claude API + Groq (fallback) | LLM-powered clinical triage and reporting |
| Graph Workflow | LangGraph | Multi-step agentic screening orchestration |

---

## Project Structure

```
backend/
  __init__.py
  app/
    __init__.py
    main.py                        # FastAPI app entry point, lifespan, middleware registration
    core/
      __init__.py
      config.py                    # Pydantic Settings — all env vars + defaults
      model_service.py             # Singleton model lifecycle, inference pipeline, KG
      auth.py                      # JWT creation, validation, RBAC dependencies
      logging_config.py            # Structured JSON log formatter, logger setup
      prediction_logger.py         # Append-only JSONL prediction audit trail
    middleware/
      __init__.py
      request_id.py                # X-Request-ID tracing + X-Response-Time-Ms
      rate_limit.py                # Per-IP token-bucket rate limiting
    routers/
      __init__.py
      health.py                    # GET /, /health, /health/model
      predict.py                   # POST /api/v1/predict, GET /api/v1/diseases
      auth.py                      # POST /api/v1/auth/token
      review.py                    # Human-in-the-loop review CRUD
      clinical.py                  # Disease info, knowledge graph, clinical reasoning
      explain.py                   # GradCAM, LIME, SHAP, IG, ELI5 endpoints
      analytics.py                 # System info + prediction analytics
      agents.py                    # Agent orchestration, screening pipeline, events
      governance.py                # Governance: drift, fairness, model cards, audit
      gate.py                      # Fundus gate v2: status, validate debug endpoints
      predict_edge.py              # Edge inference: ONNX, CoreML, quantized (Phase 3)
```

**Total:** ~2,200 lines of Python across 19 modules.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Start development server (auto-reload)
make backend
# or
PYTHONPATH=. uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload

# With GPU selection
CUDA_VISIBLE_DEVICES=2 make backend
```

API available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI schema: `http://localhost:8080/openapi.json`

---

## Core Services

### ModelService (`core/model_service.py`)

Singleton service that manages the full model lifecycle — loading, inference, and clinical reasoning.

**Lifecycle:**
1. **Startup** — `model_service.load()` called during FastAPI lifespan
2. **Serving** — Thread-safe for async request handlers
3. **Shutdown** — `model_service.unload()` releases GPU memory

**Model Loading:**
- Supports 4 architectures: **ViGNN**, SceneGraphTransformer, GraphCLIP, VisualLanguageGNN
- Loads from checkpoint at the configured `MODEL_PATH`
- Extracts `model_name`, `model_state_dict`, and `decision_thresholds` from checkpoint
- Falls back to **demo mode** with random predictions if checkpoint not found (never crashes)

**Inference Pipeline:**
1. Image preprocessing — resize to 224×224, normalize (ImageNet mean/std)
2. Forward pass — returns logits, apply sigmoid for probabilities
3. Threshold resolution — per-class learned thresholds or scalar fallback
4. Clinical reasoning — knowledge graph applies co-occurrence patterns
5. Referral priority — URGENT / HIGH / MEDIUM / LOW classification

**Disease Coverage:** 45 retinal diseases with codes (DR, ARMD, MH, CRVO, CRAO, etc.)

**Knowledge Graph:**
- `ClinicalKnowledgeGraph` with disease co-occurrence edges
- Categories: VASCULAR, DEGENERATIVE, STRUCTURAL, GLAUCOMATOUS, INFECTIOUS_IMMUNOLOGIC
- Refines raw model probabilities based on medical co-occurrence patterns
- Provides visual findings, treatment recommendations, and referral priority

### ModelExplainer (`src/models/model_explainer.py`)

Lazy-initialized singleton providing 5 explainability methods:

| Method | Library | Output |
|---|---|---|
| GradCAM / GradCAM++ / ScoreCAM | `grad-cam` | Heatmap overlay images (base64) |
| LIME | `lime` | Superpixel feature importance |
| SHAP | `shap` | Game-theoretic feature importance |
| Integrated Gradients | `captum` | Pixel-level attribution maps |
| ELI5 | Custom | Human-readable top features + natural language |

Initialized on first explainability request to avoid unnecessary GPU memory allocation.

### PredictionLogger (`core/prediction_logger.py`)

Append-only audit trail for regulatory compliance and drift detection.

- **Format:** Daily JSONL files at `logs/predictions/predictions_YYYY-MM-DD.jsonl`
- **Fields per entry:** timestamp, request_id, user, threshold, threshold_source, inference_ms, model_loaded, image dimensions, num_detected, referral_priority, top 5 predictions
- **Rotation:** Automatic daily file rotation
- **Retention:** Analytics reads last 30 days

### JWT Auth Service (`core/auth.py`)

- **Signing:** HMAC-SHA256 with configurable secret
- **Token structure:** `Header.Payload.Signature` (base64url-encoded)
- **Payload fields:** `sub` (subject), `exp` (expiry), `role` (user/admin), `iat` (issued-at)
- **RBAC:** Two roles — `user` and `admin`; admin bypasses all role checks
- **Toggle:** `AUTH_ENABLED=false` (default) skips auth for development

### Additional Services (initialized during lifespan)

These services are initialized in `main.py` lifespan and are opt-in via environment variables:

| Service | Module | Env Toggle | Purpose |
|---------|--------|-----------|---------|
| AgentOrchestrator | `src/agents/orchestrator` | Always (non-fatal) | LangGraph multi-agent screening pipeline |
| ReviewGate | `routers/review.py` | Always | Human-in-the-loop clinical review queue |
| HealthMonitor | `src/monitoring/health` | Always | Latency tracking, SLA compliance, throughput |
| GateMonitor | `src/monitoring/gate_monitor` | `FUNDUS_GATE__ENABLED` | Gate pass/reject rates, disagreements, alerting |
| LearnedFundusGate | `src/data/fundus_gate_learned` | `FUNDUS_GATE__ENABLED` | MobileNetV3-Small binary classifier for gate v2 |
| DriftDetector | `core/drift_detector` | `DRIFT__ENABLED` | PSI + KS test + NannyML + Evidently drift detection |
| ActiveLearningLoop | `core/active_learning` | `ACTIVE_LEARNING_LOOP__ENABLED` | Review → LoRA fine-tune → MLflow registration |
| MLflowRegistry | `core/mlflow_registry` | `MLFLOW__ENABLED` | Model versioning, staging/production promotion |
| Telemetry | `core/telemetry` | `TELEMETRY__ENABLED` | OpenTelemetry distributed tracing + metrics |

All optional services are wrapped in try/except during startup — failure is non-fatal and logged as a warning.

---

## Middleware Stack

Middleware executes in reverse registration order (last registered = first executed):

| Order | Middleware | Purpose | Headers |
|---|---|---|---|
| 1 | Security Headers | XSS, clickjacking, HSTS protection | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`, `Strict-Transport-Security` (prod) |
| 2 | Rate Limiter | Token-bucket per client IP (60 req/min) | `X-RateLimit-Limit`, `X-RateLimit-Remaining` |
| 3 | Request ID | UUID tracing + response time measurement | `X-Request-ID`, `X-Response-Time-Ms` |
| 4 | GZIP | Compress responses > 1 KB | `Content-Encoding: gzip` |
| 5 | CORS | Cross-origin access control | Standard CORS headers |

**Rate Limiting Details:**
- Algorithm: Token-bucket per client IP
- Limit: Configurable via `RATE_LIMIT_PER_MINUTE` (default 60)
- Proxy support: Reads `X-Forwarded-For` header for real client IP
- Response on exceed: `429 Too Many Requests` with `Retry-After: 60`

---

## API Reference

### Health & Monitoring

#### `GET /`
Root metadata — app name, version, GPU availability, docs link.
```bash
curl http://localhost:8080/
```
```json
{
  "app": "RetinalAI Clinical Screening Platform",
  "version": "3.0.0",
  "gpu_available": true,
  "model_loaded": true,
  "docs": "/docs"
}
```

#### `GET /health`
Basic health check for load balancers and uptime monitors.
```bash
curl http://localhost:8080/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "diseases_count": 45
}
```

#### `GET /health/model`
Detailed model performance metrics and SLA compliance.
```json
{
  "latency_p50_ms": 23.1,
  "latency_p95_ms": 38.5,
  "latency_p99_ms": 45.2,
  "throughput_rps": 12.3,
  "error_rate": 0.0,
  "total_predictions": 150,
  "sla_compliant": true
}
```

---

### Prediction

#### `POST /api/v1/predict`
Upload a retinal fundus image for multi-label disease classification.

**Input validation (5 layers):**
1. Content-type check (`image/*`)
2. File size check (max 10 MB)
3. Magic bytes validation (JPEG: `FFD8FF`, PNG: `89504E47`)
4. PIL structural verification (`Image.verify()`)
5. Minimum resolution (32×32)

**Fundus image gating (v2 fusion — 4 layers):**
1. **Structural** — format, resolution, aspect ratio, color mode
2. **Statistical** — channel histograms, red dominance, dark border, center brightness, radial sharpness
3. **Learned** — MobileNetV3-Small binary classifier (runs only if statistical passes)
4. **Fusion** — `0.6 * statistical + 0.4 * learned`, threshold 0.70, hard spatial requirement
5. **Post-inference OOD** — flags if model confidence is near-zero across all diseases

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Authorization: Bearer <token>" \
  -F "file=@retinal_image.png" \
  -F "threshold=0.5"
```

**Response:**
```json
{
  "success": true,
  "request_id": "abc-123-def",
  "predictions": [
    {
      "code": "DR",
      "name": "Diabetic Retinopathy",
      "probability": 0.87,
      "threshold": 0.45,
      "confidence": "high"
    }
  ],
  "total_detected": 3,
  "all_probabilities": {"DR": 0.87, "ARMD": 0.12, "...": "..."},
  "clinical": {
    "referral_priority": "URGENT",
    "refined_predictions": {"DR": 0.87, "CME": 0.45}
  },
  "fundus_gate": {
    "passed": true,
    "confidence": 0.95,
    "version": "v2",
    "latency_ms": 7.2,
    "statistical_confidence": 0.92,
    "learned_confidence": 0.98,
    "fusion_confidence": 0.95
  },
  "inference_ms": 27.5,
  "model_loaded": true
}
```

Every prediction is logged to `logs/predictions/` and emits a `SCAN_ANALYZED` event to the agent event bus.

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Invalid image format, too small, wrong content-type |
| `413` | File exceeds 10 MB |
| `422` | Non-fundus image rejected by gating (returns layer, reason, confidence) |
| `500` | Internal model error |

#### `GET /api/v1/diseases`
List all 45 detectable diseases with codes and full names.
```bash
curl http://localhost:8080/api/v1/diseases
```
```json
{
  "total": 45,
  "diseases": [
    {"code": "DR", "name": "Diabetic Retinopathy"},
    {"code": "ARMD", "name": "Age-related Macular Degeneration"},
    "..."
  ]
}
```

---

### Authentication

#### `POST /api/v1/auth/token`
Exchange an API key for a JWT access token.

```bash
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-key"}'
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

- **Dev mode** (`AUTH_ENABLED=false`): Returns an admin token immediately without validation
- **Production** (`AUTH_ENABLED=true`): Validates API key against `JWT_SECRET`

Use the token in subsequent requests:
```bash
curl -H "Authorization: Bearer eyJ..." http://localhost:8080/api/v1/predict ...
```

---

### Human-in-the-Loop Review

#### `GET /api/v1/review/pending`
Fetch pending human review requests.
```bash
curl "http://localhost:8080/api/v1/review/pending?priority=urgent"
```
- Optional query param: `priority` (urgent, high, medium, low)
- Returns: list of reviews with `request_id`, `prediction_id`, `reason`, `priority`

#### `POST /api/v1/review/{request_id}/resolve`
Resolve a pending review with clinician decision.
```bash
curl -X POST http://localhost:8080/api/v1/review/req-123/resolve \
  -H "Content-Type: application/json" \
  -d '{"decision": "confirmed", "notes": "Agrees with DR finding"}'
```
- Decision values: `confirmed`, `rejected`, `modified`, `escalated`

#### `GET /api/v1/review/stats`
Review queue statistics.
```json
{
  "total": 42,
  "pending": 5,
  "resolved": 37,
  "by_priority": {"urgent": 2, "high": 3, "medium": 10, "low": 27}
}
```

See [Governance](11-governance.md) for the full human review framework.

---

### Clinical Reasoning

#### `GET /api/v1/clinical/disease-info/{code}`
Get clinical information for a specific disease.
```bash
curl http://localhost:8080/api/v1/clinical/disease-info/DR
```
```json
{
  "code": "DR",
  "name": "Diabetic Retinopathy",
  "info_available": true,
  "severity": 3,
  "category": "VASCULAR",
  "description": "Damage to retinal blood vessels caused by diabetes...",
  "risk_factors": ["Diabetes", "Hypertension", "High cholesterol", "Pregnancy"],
  "treatment": ["Glycemic control", "Laser photocoagulation", "Anti-VEGF injections", "Vitrectomy"],
  "urgency": "Immediate referral within 24-48h"
}
```

**Severity scale:** 1 (low) → 2 (moderate) → 3 (high/sight-threatening)

**Categories:** VASCULAR, DEGENERATIVE, STRUCTURAL, GLAUCOMATOUS, INFECTIOUS_IMMUNOLOGIC

#### `GET /api/v1/clinical/disease-info`
Get clinical info for all 45 diseases in a single request.

#### `GET /api/v1/clinical/knowledge-graph`
Knowledge graph data for frontend visualization.
```json
{
  "diseases": 45,
  "edges": 120,
  "categories": {"VASCULAR": ["DR", "CRVO", "..."], "...": "..."},
  "relationships": [{"source": "DR", "target": "CME", "type": "co-occurrence"}],
  "severity": {"DR": 3, "ARMD": 2, "...": "..."},
  "prevalence": {"DR": 0.15, "...": "..."},
  "disease_names": {"DR": "Diabetic Retinopathy", "...": "..."}
}
```

#### `POST /api/v1/clinical/explain-reasoning`
Explain how clinical reasoning adjusts raw model predictions.
```bash
curl -X POST http://localhost:8080/api/v1/clinical/explain-reasoning \
  -H "Content-Type: application/json" \
  -d '{"DR": 0.85, "CME": 0.3, "VH": 0.15}'
```
```json
{
  "adjustments": [
    {
      "disease": "CME",
      "name": "Cystoid Macular Edema",
      "original": 0.3,
      "refined": 0.42,
      "boost": 0.12,
      "reason": "DR detected with high confidence - CME frequently co-occurs with diabetic retinopathy"
    }
  ],
  "referral_priority": "URGENT",
  "visual_findings": ["..."],
  "treatment_recommendations": ["..."],
  "detected_count": 2
}
```

- Input: dict of disease codes → probabilities (0.0–1.0, max 50 entries)
- The knowledge graph boosts/suppresses probabilities based on medical co-occurrence patterns

---

### Explainability

All explainability endpoints accept an image upload and return method-specific results.
Explainer is lazy-initialized on first request to conserve GPU memory.

#### `POST /api/v1/explain/gradcam`
Generate GradCAM class activation heatmap overlays.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `target_class` | int | auto | Class index (auto-selects top prediction) |
| `method` | string | `GradCAM` | Variant: `GradCAM`, `GradCAMPlusPlus`, `ScoreCAM` |
| `top_k` | int | 3 | Number of top classes to explain (1–10) |

```bash
curl -X POST http://localhost:8080/api/v1/explain/gradcam \
  -F "file=@retinal.png" -F "top_k=3"
```

Returns base64-encoded heatmap images per class with disease code, name, and probability.

#### `POST /api/v1/explain/lime`
Generate LIME superpixel explanations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `top_k` | int | 3 | Classes to explain (1–5) |
| `num_samples` | int | 300 | Perturbation samples (50–2000, higher = slower but more accurate) |
| `num_features` | int | 10 | Number of superpixels (3–30) |

#### `POST /api/v1/explain/shap`
Generate SHAP feature importance explanations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `top_k` | int | 3 | Classes to explain (1–5) |

#### `POST /api/v1/explain/integrated-gradients`
Generate Integrated Gradients pixel-level attribution maps (via Captum).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `top_k` | int | 2 | Classes to explain (1–5) |
| `n_steps` | int | 25 | Integration steps (5–100) |

#### `POST /api/v1/explain/eli5`
Generate human-readable ELI5 explanations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `top_k` | int | 3 | Classes to explain (1–5) |
| `top_features` | int | 10 | Features to show (3–20) |

#### `POST /api/v1/explain/comprehensive`
Run all available methods in one call — GradCAM heatmaps + clinical insights + uncertainty metrics.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (JPEG/PNG) |
| `top_k` | int | 3 | Classes to explain (1–10) |

Returns combined results with `gradcam_heatmaps`, `explainability` (LIME/SHAP/IG data), `clinical_insights`, and `uncertainty` metrics.

#### `GET /api/v1/explain/available`
Check which methods are available in the current environment.
```json
{
  "model_loaded": true,
  "methods": {
    "gradcam": {"available": true, "description": "Gradient-weighted Class Activation Mapping..."},
    "integrated_gradients": {"available": true, "description": "Attribution method from Captum..."},
    "shap": {"available": true, "description": "SHapley Additive exPlanations..."},
    "lime": {"available": true, "description": "Local Interpretable Model-agnostic Explanations..."},
    "eli5": {"available": true, "description": "Explain Like I'm 5..."}
  }
}
```

---

### Analytics & System Info

#### `GET /api/v1/system/info`
Comprehensive deployment and capability information for the platform dashboard.
```json
{
  "platform": {
    "name": "RetinalAI Clinical Screening Platform",
    "version": "3.0.0",
    "environment": "production",
    "region": "default",
    "regulatory_mode": "research"
  },
  "model": {
    "name": "vignn",
    "loaded": true,
    "num_classes": 45,
    "diseases_covered": 45,
    "knowledge_graph_edges": 120,
    "threshold_source": "per_class"
  },
  "infrastructure": {
    "python_version": "3.12.0",
    "pytorch_version": "2.2.0",
    "cuda_available": true,
    "cuda_version": "12.2",
    "gpu": "NVIDIA RTX A6000",
    "gpu_memory": "48.0 GB",
    "device": "cuda:0"
  },
  "capabilities": {
    "explainability_methods": ["GradCAM", "LIME", "SHAP", "Integrated Gradients", "ELI5"],
    "clinical_reasoning": true,
    "knowledge_graph": true,
    "human_review": true,
    "audit_trail": true,
    "drift_detection": true
  },
  "compliance": {
    "eu_ai_act": "conformity_ready",
    "fda_samd": "pre_submission",
    "data_governance": true,
    "model_cards": true,
    "fairness_evaluation": true,
    "prediction_logging": true
  }
}
```

#### `GET /api/v1/analytics/summary`
Prediction analytics aggregated from the last 30 days of audit logs.
```json
{
  "total_scans": 1523,
  "today_scans": 47,
  "avg_inference_ms": 28.3,
  "referral_distribution": {"URGENT": 12, "HIGH": 45, "MEDIUM": 320, "LOW": 1146},
  "top_detected_diseases": [
    {"code": "DR", "count": 234},
    {"code": "ARMD", "count": 156}
  ],
  "daily_volumes": [
    {"date": "2026-04-27", "scans": 52},
    {"date": "2026-04-28", "scans": 47}
  ]
}
```

---

### Agent Orchestration

The agents subsystem provides autonomous clinical screening via a LangGraph multi-step workflow.

#### `GET /api/v1/agents/status`
Get status of all autonomous agents (monitor, governance, etc.).

#### `GET /api/v1/agents/events`
Query agent event history.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event_type` | string | all | Filter by event type (e.g., `SCAN_ANALYZED`) |
| `source` | string | all | Filter by event source |
| `limit` | int | 50 | Max events to return |

```bash
curl "http://localhost:8080/api/v1/agents/events?event_type=SCAN_ANALYZED&limit=10"
```

#### `GET /api/v1/agents/compliance`
Generate an on-demand compliance report from the governance agent. Returns 503 if governance agent is not available.

#### `GET /api/v1/agents/tools`
List all tools available across all agents.
```json
{
  "agents": {
    "monitor": ["check_model_health", "check_drift", "..."],
    "governance": ["generate_compliance_report", "audit_prediction", "..."]
  },
  "total_tools": 12
}
```

#### `POST /api/v1/agents/screen`
**Full agentic screening pipeline** — the LLM-powered alternative to `/api/v1/predict`.

Orchestrates a 6-node LangGraph workflow:
```
classify → triage (Claude) → reason (KG) → explain (conditional) → review (conditional) → report (Claude)
```

```bash
curl -X POST http://localhost:8080/api/v1/agents/screen \
  -F "file=@retinal.png" \
  -F "threshold=0.5"
```

**Response:**
```json
{
  "success": true,
  "scan_id": "a1b2c3d4",
  "report": {
    "predictions": [...],
    "triage": {"decision": "refer", "reasoning": "..."},
    "clinical_narrative": "Claude-generated clinical summary...",
    "explainability": {...},
    "review_status": "pending",
    "referral_priority": "URGENT"
  }
}
```

**LLM Fallback Chain:** Claude → Groq → deterministic rules (never fails).

#### `GET /api/v1/agents/graph/info`
Describe the LangGraph workflow topology.
```json
{
  "framework": "LangGraph + Multi-LLM",
  "graph_nodes": ["classify", "triage", "reason", "explain", "review", "report"],
  "conditional_edges": {
    "reason → explain|review|report": "3-way branch based on triage decisions",
    "explain → review|report": "2-way branch after explainability"
  },
  "llm_nodes": ["triage", "report"],
  "deterministic_nodes": ["classify", "reason", "explain", "review"],
  "llm_available": true,
  "active_provider": "claude",
  "active_model": "claude-sonnet-4-20250514",
  "fallback_chain": ["claude", "groq", "deterministic_rules"]
}
```

---

### Governance & Compliance

#### `GET /api/v1/governance/drift`
Current drift detection status and history (PSI, KS test results, alerts).

#### `GET /api/v1/governance/active-learning-stats`
Active learning queue size, fine-tune history, and retraining status.

#### `GET /api/v1/governance/model-registry`
MLflow model registry status — registered models, stages, versions.

#### `GET /api/v1/governance/fairness`
Fairness dashboard with demographic subgroup performance breakdowns.

#### `GET /api/v1/governance/fairness/history`
Historical fairness evaluation results over time.

#### `GET /api/v1/governance/model-card`
Current model card in JSON or Markdown format. Accepts `?format=markdown` query param.

#### `GET /api/v1/governance/audit`
Query the immutable audit log. Supports filtering by event type and date range.

#### `GET /api/v1/governance/audit/integrity`
Verify audit trail integrity using SHA-256 hash chain validation.
```json
{
  "integrity": "verified",
  "total_events": 1523,
  "chain_valid": true,
  "last_verified": "2026-04-29T10:00:00Z"
}
```

---

### Fundus Gate

#### `GET /api/v1/gate/status`
Gate configuration, runtime statistics, and learned model status.
```json
{
  "gate_version": "v2",
  "enabled": true,
  "learned_model_loaded": true,
  "config": {
    "learned_weight": 0.4,
    "min_confidence": 0.70,
    "visual_evidence": false
  },
  "stats": {
    "total_checked": 150,
    "passed": 142,
    "rejected": 8,
    "pass_rate": 0.947
  }
}
```

#### `POST /api/v1/gate/validate`
Debug endpoint — runs full gate analysis on an uploaded image but **always returns 200** (never 422). Includes visual evidence for debugging.
```bash
curl -X POST http://localhost:8080/api/v1/gate/validate -F "file=@image.png"
```

#### `GET /health/gate`
Gate-specific health metrics for monitoring dashboards.
```json
{
  "passed": 142,
  "rejected": 8,
  "pass_rate": 0.947,
  "latency_p50_ms": 5.2,
  "latency_p95_ms": 7.8,
  "latency_p99_ms": 9.1,
  "alert": false
}
```

---

### Edge Inference (Phase 3)

Optimized inference endpoints for deployment on resource-constrained devices. Requires `EDGE__ONNX_ENABLED=true` (or coreml/quantized equivalents).

#### `POST /api/v1/predict/onnx`
ONNX Runtime inference — fastest CPU inference path.

#### `POST /api/v1/predict/coreml`
Core ML inference — optimized for Apple Silicon (M-series Macs, iOS devices).

#### `POST /api/v1/predict/quantized`
INT8/FP16 quantized inference — smallest model footprint.

#### `GET /api/v1/predict/edge/status`
Loaded edge model formats and their configuration.
```json
{
  "onnx": {"loaded": true, "path": "models/export/model.onnx"},
  "coreml": {"loaded": false, "reason": "EDGE__COREML_ENABLED=false"},
  "quantized": {"loaded": true, "dtype": "int8"}
}
```

---

## Authentication & Authorization

### Token Flow

```
1. Client sends API key       →  POST /api/v1/auth/token  {"api_key": "..."}
2. Server validates key       →  Returns JWT {access_token, expires_in}
3. Client includes JWT        →  Authorization: Bearer <token>
4. Server validates on each   →  get_current_user() dependency
   request
```

### Role-Based Access Control

| Role | Permissions |
|---|---|
| `user` | Access prediction, disease info, review (read-only) |
| `admin` | All user permissions + resolve reviews, agent control, system info |

RBAC is enforced via the `require_role(role)` FastAPI dependency.

### Development Mode

When `AUTH_ENABLED=false` (default):
- `get_current_user()` returns `None` — all endpoints are accessible
- `POST /api/v1/auth/token` returns an admin token without validation
- No `Authorization` header required

### Production Mode

When `AUTH_ENABLED=true`:
- All protected endpoints require a valid JWT
- Token expiry is enforced (default 3600s)
- Invalid/expired tokens return `401 Unauthorized`
- Insufficient role returns `403 Forbidden`

---

## Security

### Transport Security
- **HSTS** enabled in production (`Strict-Transport-Security: max-age=31536000; includeSubDomains`)
- **CORS** configured with specific allowed origins (localhost + HuggingFace)

### Response Headers
| Header | Value | Purpose |
|---|---|---|
| `X-Content-Type-Options` | `nosniff` | Prevent MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Prevent clickjacking |
| `X-XSS-Protection` | `1; mode=block` | XSS filter |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limit referrer leakage |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | Restrict browser features |

### Input Validation
- Image magic bytes verification (prevents disguised file uploads)
- File size limits (10 MB max)
- PIL structural verification before processing
- Fundus image gating rejects non-retinal images
- Request body size limits via FastAPI
- Rate limiting (60 req/min per IP) prevents abuse

### Secrets Management
- JWT secret via environment variable (`JWT_SECRET`)
- API keys via environment variables (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`)
- Never hardcoded — defaults are development-only placeholders

---

## Configuration Reference

All settings are managed via environment variables or `.env` file, loaded by Pydantic Settings.

### Application

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `RetinalAI Clinical Screening Platform` | Application display name |
| `APP_VERSION` | `3.0.0` | Semantic version |
| `DEBUG` | `false` | Enable debug mode (auto-reload) |
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `DEPLOYMENT_REGION` | `default` | Deployment region identifier |

### Server

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8080` | Port |
| `CORS_ORIGINS` | `["http://localhost:3000", ...]` | Allowed CORS origins (JSON array) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |
| `MAX_UPLOAD_SIZE` | `10485760` | Max upload size in bytes (10 MB) |

### Model

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/model_vignn_rank1.pth` | Path to model checkpoint |
| `MODEL_NAME` | `vignn` | Model architecture name |
| `NUM_CLASSES` | `45` | Number of disease classes |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |

### Authentication

| Variable | Default | Description |
|---|---|---|
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `JWT_SECRET` | `change-me-in-production-use-env-var` | HMAC signing secret |
| `JWT_EXPIRY_SECONDS` | `3600` | Token lifetime (seconds) |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` (human-readable) |
| `PREDICTION_LOG_DIR` | `logs/predictions` | Prediction audit log directory |

### Explainability

| Variable | Default | Description |
|---|---|---|
| `EXPLAIN_GRADCAM_ENABLED` | `true` | Enable GradCAM |
| `EXPLAIN_LIME_DEFAULT_SAMPLES` | `300` | LIME perturbation samples |
| `EXPLAIN_SHAP_ENABLED` | `true` | Enable SHAP |

### Fundus Gate V2

| Variable | Default | Description |
|---|---|---|
| `FUNDUS_GATE__ENABLED` | `true` | Enable pre-inference fundus gating |
| `FUNDUS_GATE__VERSION` | `v2` | Gate version (`v1` = statistical-only, `v2` = fusion) |
| `FUNDUS_GATE__LEARNED_WEIGHT` | `0.4` | Weight for learned model in fusion formula |
| `FUNDUS_GATE__MIN_CONFIDENCE` | `0.70` | Minimum fusion confidence to pass gate |
| `FUNDUS_GATE__MODEL_PATH` | `weights/fundus_gate.pth` | Path to learned gate weights |
| `FUNDUS_GATE__VISUAL_EVIDENCE` | `false` | Generate base64 heatmaps on rejection |
| `FUNDUS_GATE__MC_DROPOUT_SAMPLES` | `5` | MC Dropout samples for uncertainty estimation |

### Regulatory

| Variable | Default | Description |
|---|---|---|
| `REGULATORY_MODE` | `research` | `research` / `ce_marked` / `fda_cleared` |

### Agentic AI

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | Claude API key (primary LLM) |
| `ANTHROPIC_ORG_ID` | `""` | Anthropic organization ID |
| `AGENT_MODEL` | `claude-sonnet-4-20250514` | Claude model for agentic tasks |
| `GROQ_API_KEY` | `""` | Groq API key (fallback LLM) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for fallback |
| `GROQ_MAX_TOKENS` | `4096` | Groq max output tokens |
| `GROQ_TEMPERATURE` | `0.3` | Groq sampling temperature |
| `AGENT_MONITOR_INTERVAL` | `60.0` | Monitor agent cycle (seconds) |
| `AGENT_GOVERNANCE_INTERVAL` | `300.0` | Governance agent cycle (seconds) |

### Phase 1: Observability & MLOps

| Variable | Default | Description |
|---|---|---|
| `TELEMETRY__ENABLED` | `false` | Enable OpenTelemetry distributed tracing |
| `TELEMETRY__OTLP_ENDPOINT` | `http://localhost:4317` | OTEL Collector gRPC endpoint |
| `TELEMETRY__SERVICE_NAME` | `retinalai` | Service name in traces |
| `TELEMETRY__SAMPLE_RATE` | `1.0` | Trace sampling rate (0.0–1.0) |
| `MLFLOW__ENABLED` | `false` | Enable MLflow model registry |
| `MLFLOW__TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL |
| `MLFLOW__MODEL_NAME` | `retinalai-vignn` | Registered model name |
| `MLFLOW__EXPERIMENT_NAME` | `retinalai-production` | MLflow experiment name |
| `ACTIVE_LEARNING_LOOP__ENABLED` | `false` | Enable active learning closed loop |
| `ACTIVE_LEARNING_LOOP__RETRAIN_THRESHOLD` | `150` | Corrections before retraining |
| `ACTIVE_LEARNING_LOOP__QUEUE_DIR` | `data/active_learning` | Queue directory |
| `DRIFT__ENABLED` | `true` | Enable drift detection (on by default) |
| `DRIFT__CHECK_INTERVAL` | `100` | Check every N predictions |
| `DRIFT__NANNYML_ENABLED` | `false` | Enable NannyML integration |
| `DRIFT__EVIDENTLY_ENABLED` | `false` | Enable Evidently integration |
| `DRIFT__ALERT_WEBHOOK_URL` | `""` | Webhook for drift alerts |

### Phase 2: Scalability & Security

| Variable | Default | Description |
|---|---|---|
| `RAY__ENABLED` | `false` | Enable Ray Serve dynamic batching |
| `RAY__SERVE_URL` | `http://localhost:8000` | Ray Serve dashboard URL |
| `CANARY__ENABLED` | `false` | Enable canary release routing |
| `CANARY__CANARY_WEIGHT` | `0.0` | Traffic weight for canary version (0.0–1.0) |
| `CANARY__STICKY_SESSIONS` | `true` | Sticky session routing for canary |
| `CIRCUIT_BREAKER__FAILURE_THRESHOLD` | `5` | Failures before circuit opens |
| `CIRCUIT_BREAKER__RECOVERY_TIMEOUT_S` | `60.0` | Seconds before half-open |
| `KAFKA__ENABLED` | `false` | Enable Kafka audit streaming |
| `KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `ICEBERG__ENABLED` | `false` | Enable Apache Iceberg audit tables |
| `MTLS__ENABLED` | `false` | Enable mutual TLS between services |

### Phase 3: Governance & Edge

| Variable | Default | Description |
|---|---|---|
| `EDGE__ONNX_ENABLED` | `false` | Enable ONNX Runtime inference endpoint |
| `EDGE__COREML_ENABLED` | `false` | Enable Core ML inference endpoint |
| `EDGE__QUANTIZED_ENABLED` | `false` | Enable INT8/FP16 quantized inference |
| `FAIRNESS__ENABLED` | `false` | Enable fairness dashboard endpoint |
| `MODEL_CARD__AUTO_GENERATE` | `false` | Auto-generate model cards on MLflow promotion |

### Phase 4: Future-Proofing

| Variable | Default | Description |
|---|---|---|
| `RESILIENCE__ENABLED` | `false` | Enable graceful degradation + health-aware routing |
| `RESILIENCE__HEALTH_CHECK_INTERVAL_S` | `30.0` | Health check frequency |
| `MULTIMODAL__ENABLED` | `false` | Enable multi-modal fusion (fundus + OCT) |
| `MULTIMODAL__FUSION_STRATEGY` | `concatenation` | `concatenation` or `cross_attention` |
| `FEDERATED__ENABLED` | `false` | Enable federated learning client |
| `FEDERATED__FRAMEWORK` | `flower` | `flower` or `nvflare` |
| `FEDERATED__DP_ENABLED` | `false` | Enable differential privacy |

> **Note:** All Phase 1-4 features are opt-in and disabled by default. Enabling a feature when its infrastructure is unavailable is non-fatal — the service logs a warning and continues.

---

## Logging & Audit Trail

### Structured JSON Logging (`core/logging_config.py`)

All application logs are emitted as single-line JSON for aggregation systems (ELK, CloudWatch, Datadog).

**Log format:**
```json
{
  "timestamp": "2026-04-28T10:30:00.123456Z",
  "level": "INFO",
  "logger": "backend.app.routers.predict",
  "message": "Prediction completed",
  "module": "predict",
  "function": "predict",
  "line": 82,
  "request_id": "abc-123-def",
  "latency_ms": 27.5,
  "status_code": 200
}
```

- **Timestamp:** ISO 8601 UTC
- **Extra fields:** `request_id`, `user`, `latency_ms`, `endpoint`, `status_code` (when present)
- **Exceptions:** Full traceback included in JSON
- **Noisy loggers suppressed:** `uvicorn.access`, `watchfiles`

### Prediction Audit Trail (`core/prediction_logger.py`)

Separate from application logs — purpose-built for regulatory audit and drift detection.

**Storage:** `logs/predictions/predictions_YYYY-MM-DD.jsonl`

**Entry format:**
```json
{
  "timestamp": "2026-04-28T10:30:00Z",
  "request_id": "abc-123",
  "user": "clinician@hospital.org",
  "threshold": 0.45,
  "threshold_source": "per_class",
  "inference_ms": 27.5,
  "model_loaded": true,
  "image_size": [2048, 1536],
  "num_detected": 3,
  "referral_priority": "URGENT",
  "top_predictions": [
    {"code": "DR", "probability": 0.87},
    {"code": "CME", "probability": 0.45}
  ]
}
```

---

## Error Handling

### Global Exception Handler

All unhandled exceptions are caught by the global handler in `main.py`:
```json
{
  "detail": "Internal server error. Please try again."
}
```
The full error is logged (with traceback) but never exposed to the client.

### HTTP Error Codes

| Code | Meaning | When |
|---|---|---|
| `400` | Bad Request | Invalid image format, malformed input |
| `401` | Unauthorized | Missing or invalid JWT (when `AUTH_ENABLED=true`) |
| `403` | Forbidden | Insufficient role for endpoint |
| `413` | Payload Too Large | File exceeds `MAX_UPLOAD_SIZE` |
| `422` | Unprocessable Entity | Non-fundus image rejected by gating |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Unhandled exception |
| `503` | Service Unavailable | Model not loaded, agent orchestrator not initialized |

### Structured Error Responses

Fundus gate rejections include diagnostic details:
```json
{
  "detail": {
    "error": "non_fundus_image",
    "message": "Image lacks retinal fundus characteristics (red channel not dominant)",
    "confidence": 0.23,
    "layer": "statistical",
    "checks": {"red_dominance": false, "dark_border": false, "center_brightness": true}
  }
}
```

---

## Deployment

### Docker (GPU)

```bash
# GPU build
docker compose up -d api

# CPU-only build
docker compose --profile cpu up -d api-cpu
```

### Hugging Face Spaces (Docker)

The HF Spaces deployment (`Dockerfile.hf`) runs the full stack (backend + frontend + nginx) in a single container via supervisord, optimized for CPU-only free-tier hardware.

**Architecture:**
```
supervisord (PID 1)
  ├── nginx           :7860  (reverse proxy — static assets + routing)
  ├── uvicorn/FastAPI  :8080  (backend API — internal only)
  └── node server.js   :3000  (Next.js standalone — internal only)
```

**Key differences from GPU deployment:**
| Aspect | GPU (`Dockerfile`) | HF Spaces (`Dockerfile.hf`) |
|--------|-------------------|----------------------------|
| Base image | `nvidia/cuda:12.1.1` | `python:3.11-slim-bookworm` |
| PyTorch | CUDA 12.1 (`whl/cu121`) | CPU-only (`whl/cpu`) |
| Port | 8080 (API only) | 7860 (nginx reverse proxy) |
| Frontend | Separate (`make frontend`) | Built into Docker image |
| Process manager | Single uvicorn | supervisord (3 processes) |
| Device | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=-1`, `DEVICE=cpu` |

**Supervisord configuration** (`supervisord.conf`):
- Environment variables are hardcoded (not `%(ENV_*)s` expansion) because HF Spaces does not inject standard CUDA env vars
- Backend binds to `127.0.0.1:8080` (internal, proxied by nginx)
- Frontend binds to `127.0.0.1:3000` (internal, proxied by nginx)
- All processes set `autorestart=true` with 3 retries

**Nginx routing** (`nginx.conf`):
| Path | Destination | Purpose |
|------|-------------|---------|
| `/_next/static/*` | `/srv/nextjs/` (filesystem) | Pre-built Next.js static assets (365d cache) |
| `/api/*` | `proxy_pass :8080` | Backend API passthrough |
| `/health` | `proxy_pass :8080` | Health check (used by HF Spaces HEALTHCHECK) |
| `/docs` | `proxy_pass :8080` | Swagger UI |
| `/openapi.json` | `proxy_pass :8080` | OpenAPI schema |
| `/*` | `proxy_pass :3000` | Next.js pages and SSR |

**Deploy to HF Spaces:**
```bash
# Automated (clones Space, syncs files, pushes)
make deploy-hf
# or
HF_TOKEN=hf_xxx bash scripts/deploy_hf.sh

# Local test before pushing
docker compose --profile hf up --build
```

**Troubleshooting HF Spaces:**
- Check build logs at `https://huggingface.co/spaces/<user>/<space>`
- Runtime status: `curl https://huggingface.co/api/spaces/<user>/<space>/runtime`
- Common failures: supervisor env var expansion (`%(ENV_X)s` fails if `X` is not set), CUDA base image on CPU hardware, `short_description` > 60 chars in README frontmatter

### Manual (Development)

```bash
# Install
uv sync

# Run with auto-reload
PYTHONPATH=. uv run uvicorn backend.app.main:app \
  --host 0.0.0.0 --port 8080 --reload

# Run with specific GPU
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run uvicorn backend.app.main:app \
  --host 0.0.0.0 --port 8080
```

### Environment Checklist (Production)

| Item | Action |
|---|---|
| `AUTH_ENABLED` | Set to `true` |
| `JWT_SECRET` | Set to a strong random secret (not the default) |
| `ENVIRONMENT` | Set to `production` |
| `DEBUG` | Set to `false` |
| `CORS_ORIGINS` | Restrict to actual frontend domains |
| `ANTHROPIC_API_KEY` | Set for agentic screening |
| `LOG_FORMAT` | Keep `json` for log aggregation |
| `RATE_LIMIT_PER_MINUTE` | Tune based on expected traffic |
| SSL/TLS | Terminate at reverse proxy (nginx, Caddy, ALB) |
| Model checkpoint | Ensure `MODEL_PATH` points to production weights |
| `FUNDUS_GATE__ENABLED` | Set to `true` for pre-inference safety gating |
| `FUNDUS_GATE__MODEL_PATH` | Ensure learned gate weights exist at configured path |

---

## Key Design Decisions

1. **Lifespan management** — Model loads once on startup, releases on shutdown (not per-request). GPU memory is allocated exactly once.

2. **Singleton services** — `model_service` and `_explainer` are global singletons, thread-safe for async request handlers.

3. **Lazy explainer initialization** — ModelExplainer is only created when the first explainability endpoint is called, avoiding unnecessary GPU allocation for prediction-only workloads.

4. **Clinical reasoning layer** — Knowledge graph refines raw model probabilities using medical co-occurrence patterns, producing more clinically meaningful results.

5. **Demo mode** — If no model checkpoint is found, the API returns random predictions and never crashes. Useful for frontend development and CI testing.

6. **4-layer fundus gating (v2)** — Non-fundus images are rejected before inference via a fusion of statistical rules and a learned MobileNetV3-Small classifier. Visual evidence (GradCAM, heatmaps) supports explainable rejection for regulatory compliance.

7. **Prediction audit trail** — Every inference is logged to append-only JSONL files for regulatory compliance, drift detection, and analytics.

8. **Auth toggle** — Authentication is disabled by default for development, avoiding friction during local development while enforcing security in production.

9. **LLM fallback chain** — Agentic screening falls back from Claude → Groq → deterministic rules, ensuring the system always produces results regardless of API availability.

10. **Event-driven agents** — Predictions emit `SCAN_ANALYZED` events to the agent bus, enabling autonomous monitoring, governance, and alerting without coupling to the prediction pipeline.

11. **Structured JSON logging** — All logs are machine-parseable for integration with ELK, CloudWatch, Datadog, or any log aggregation system.

12. **Security headers by default** — XSS protection, clickjacking prevention, HSTS (production), and permissions policy are applied to every response via middleware.

13. **HF Spaces single-container deployment** — Supervisord orchestrates nginx + backend + frontend in one Docker container with CPU-only PyTorch, enabling zero-config deployment on Hugging Face Spaces free tier. Hardcoded env vars avoid supervisor expansion failures on restricted platforms.

14. **Configuration hierarchy** — Nested Pydantic Settings with `__` delimiter (e.g., `FUNDUS_GATE__ENABLED`) allows fine-grained control via env vars without code changes. All features are opt-in with safe defaults.

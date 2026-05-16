# OptiscanAI Clinical Screening Platform v4.0

Offline-first retinal disease screening platform for rural Uganda. Classifies 24 retinal diseases from fundus photographs using **RETFound ViT-Large** with LoRA adapters, clinical knowledge graph reasoning, and a LangGraph agentic workflow. Designed for community health workers on mid-range Android phones with intermittent connectivity.

**Production model (v2):** Precision 0.312 (12.5x over v1), F1 0.362 (7.9x), AUC 0.888, Accuracy 95.4%.

**v4.0 additions:** On-device MobileNetV3 student model (5.2M params, INT8 ONNX), Flutter mobile app with Drift database, voice-first interface (Whisper + Piper TTS) with Luganda support, DHIS2/FHIR/DICOM integration, mobile money referral payments, federated learning with LoRA adapter exchange, and ISO 14971 risk management.

**SaaS / billing layer (Phase 6).** OptiscanAI now runs as a multi-tenant
SaaS at [www.optiscan.makstartup.com](https://www.optiscan.makstartup.com)
with a 4-tier subscription model (Free · Clinician · Practice · Health
System), built-in auth (JWT + magic link + refresh-token rotation), monthly
scan quotas with paywall + upsell UX, team-seat management, and four
payment rails (Stripe + MTN MoMo + Airtel Money + Flutterwave). Marketing
site, `/pricing`, `/legal/{privacy,terms}`, and a superuser admin
ops view at `/app/admin/webhooks` all ship in the same Next.js app.
The full architecture and runbook is in
[docs/23-billing-platform.md](docs/23-billing-platform.md). All of it is
**opt-in** — flip `BILLING__ENABLED=false` and the layer goes dormant,
preserving the original on-prem / research deployment story.

## Architecture (2026)

```
RetinalFoundationHybrid — Unified Production Model
===================================================
  Input (3x224x224)
    -> RETFound ViT-Large (304M params, MAE-pretrained on 1.6M retinal images)
    -> LoRA Adapters (rank 16, 2.4M trainable params)
    -> Lightweight Graph Reasoning Head (SparseTopK + Disease Prototypes)
    -> Mixture-of-Experts Router (9 disease category experts)
    -> Uncertainty Quantification (MC Dropout + 3 Ensemble Heads)
    -> ClinicalKnowledgeGraph (Uganda-specific epidemiology)
    -> Output: 48 disease predictions + uncertainty + clinical reasoning

Target Performance:
  - Multi-label AUC: 0.90-0.96
  - p99 latency: <12ms (A100 INT8)
  - Model size: <75MB (INT8 quantized)
```

```
Stack:
  Backend:     FastAPI + UV + JWT Auth       (backend/)
  Frontend:    Next.js 16 + Bun             (frontend/)
  Training:    PyTorch DDP + LoRA           (src/ + train.py)
  Model:       RetinalFoundationHybrid      (src/models/retinal_foundation_hybrid.py)
  Legacy:      ViGNN, GraphCLIP, SceneGraphTransformer, VisualLanguageGNN
  Data:        RFMiD dataset (3,200 retinal fundus images, 45 disease classes)
  Pipeline:    DVC (data versioning + reproducible stages)
  Observability: OpenTelemetry + Jaeger + Prometheus     (Phase 1)
  Registry:    MLflow 3.0 Model Registry                 (Phase 1)
  Active ML:   Closed-loop: review -> LoRA fine-tune     (Phase 1)
  Drift:       PSI + KS + NannyML + Evidently            (Phase 1)
  Serving:     Ray Serve (dynamic batching, canary) + vLLM Qwen3-8B-AWQ runtime
  Security:    mTLS + SBOM + Kafka+Iceberg audit         (Phase 2)
  Resilience:  Circuit breakers + chaos engineering       (Phase 2)
  Governance:  Auto model cards + fairness dashboard     (Phase 3)
  Edge:        ONNX + CoreML + INT8 inference endpoints  (Phase 3)
  Safety:      Fundus gate v2 (statistical + learned MobileNetV3 fusion, <12ms p99)
  Agents:      LangGraph 7-node pipeline (classify->extract_history->triage->reason->explain->review->report)
  Mobile:      Flutter + Drift + Riverpod + ONNX Runtime (offline-first, Phase 5)
  Voice:       Whisper-tiny ASR + Piper TTS + Silero VAD (Luganda + English, Phase 5)
  Uganda:      DHIS2 + MTN MoMo + Airtel Money + Africa's Talking SMS/USSD (Phase 5)
  Clinical:    FHIR R4 + DICOM + bilingual referral letters (Phase 5)
  Federated:   Flower LoRA-only exchange + Opacus DP-SGD (Phase 5)
  Privacy:     PDP Act 2019 consent + data minimization + cross-border controls
  Deploy:      Docker (GPU/CPU), HF Spaces (supervisord + nginx), K8s
  CI/CD:       GitHub Actions (lint, test, bias audit, bundle size, faithfulness gates)
```

## Quick Start

```bash
# Install (core)
uv sync && cd frontend && bun install && cd ..

# Install with production features (Phase 1-3)
pip install -e ".[observability,drift-detection,ray-serve,edge]"

# Development (backend:8080 + frontend:3000)
make dev

# Run tests (198 tests)
make test

# Train on 8 GPUs
make train

# Export to all formats (ONNX, TorchScript, INT8, FP16)
make export-all

# Full MLOps pipeline (validate -> train -> export -> model card)
make mlops-pipeline
```

### Enable 2026 Production Features

```bash
# Phase 1: Start observability + MLflow stack
make up-phase1
# API automatically connects to OTEL Collector + MLflow
# Jaeger UI: http://localhost:16686 | MLflow UI: http://localhost:5000

# Phase 2: Add Ray Serve + Kafka
make up-phase2

# Full stack (all phases)
make up-full

# Teardown
make down-full
```

All features are **opt-in via environment variables** and disabled by default. See [Migration Guide](docs/18-migration-guide.md) for step-by-step activation.

### Deploy to Hugging Face Spaces

```bash
# Automated deployment (requires HF_TOKEN)
make deploy-hf

# Local test (full stack: backend + frontend + nginx on :7860)
docker compose --profile hf up --build
```

**Live Space**: [mpairwe49-retinal-screening.hf.space](https://mpairwe49-retinal-screening.hf.space)

The HF Spaces deployment uses `Dockerfile.hf` — a CPU-optimized single container with supervisord orchestrating nginx (:7860), FastAPI backend (:8080), and Next.js standalone (:3000). See [Frontend Setup](docs/07-frontend-setup.md#deployment-modes) for architecture details.

## Project Structure

```
.
├── backend/
│   └── app/
│       ├── core/
│       │   ├── config.py              Nested Pydantic settings (20+ feature sections)
│       │   ├── model_service.py       Model lifecycle + OTEL tracing + drift hooks
│       │   ├── telemetry.py           OpenTelemetry setup (Phase 1)
│       │   ├── mlflow_registry.py     MLflow 3.0 model registry (Phase 1)
│       │   ├── active_learning.py     Closed-loop active learning (Phase 1)
│       │   ├── drift_detector.py      Enhanced drift detection (Phase 1)
│       │   ├── mtls.py                Mutual TLS configuration (Phase 2)
│       │   ├── audit_logger.py        Kafka + Iceberg audit logs (Phase 2)
│       │   ├── model_card_generator.py  Auto model cards (Phase 3)
│       │   ├── multi_modal_fusion.py  Fundus + OCT + metadata fusion (Phase 4)
│       │   ├── federated_client.py    Flower / NVFlare client (Phase 4)
│       │   ├── graceful_degradation.py  Fallback chains (Phase 4)
│       │   ├── auth.py                JWT authentication
│       │   ├── prediction_logger.py   JSONL prediction logging
│       │   ├── logging_config.py      Structured JSON logging
│       │   ├── voice_pipeline.py      VAD + ASR + TTS orchestration (Phase 5)
│       │   ├── asr_engine.py          Whisper-tiny streaming ASR (Phase 5)
│       │   ├── tts_engine.py          Piper TTS with barge-in (Phase 5)
│       │   ├── vad_engine.py          Silero VAD for speech detection (Phase 5)
│       │   ├── referral_letter.py     Bilingual referral letter generator (Phase 5)
│       │   ├── luganda/               Luganda clinical terms, code-switching, phonemes
│       │   └── privacy/               PDP Act 2019 consent + data minimization
│       ├── middleware/
│       │   ├── request_id.py          Request ID + OTEL span bridge
│       │   └── rate_limit.py          Token-bucket rate limiter
│       ├── routers/
│       │   ├── predict.py             /api/v1/predict (main inference)
│       │   ├── predict_edge.py        /api/v1/predict/onnx,coreml,quantized (Phase 3)
│       │   ├── governance.py          /api/v1/governance/* (drift, AL, fairness, model cards)
│       │   ├── review.py              /api/v1/review/* + active learning hook
│       │   ├── agents.py              /api/v1/agents/* (LangGraph screening)
│       │   ├── explain.py             /api/v1/explain/* (GradCAM, LIME, SHAP, IG, ELI5)
│       │   ├── clinical.py            /api/v1/clinical/* (KG reasoning)
│       │   ├── health.py              /health (liveness, model metrics)
│       │   ├── auth.py                /api/v1/auth/token
│       │   ├── voice.py              WebSocket /v1/voice/stream (Phase 5)
│       │   ├── dhis2.py              /api/v1/dhis2/* (patient, referral, queue)
│       │   ├── payments.py           /api/v1/payments/* (MTN MoMo, Airtel Money)
│       │   ├── sms.py                /api/v1/sms/* (Africa's Talking SMS/USSD)
│       │   ├── fhir.py               /api/v1/fhir/* (DiagnosticReport, Bundle)
│       │   └── dicom.py              /api/v1/dicom/upload (DICOM fundus extraction)
│       └── serving/
│           └── ray_serve_config.py    Ray Serve deployment (Phase 2)
│       └── integrations/
│           ├── dhis2/             DHIS2 client, auth, models, offline queue
│           ├── mobile_money/      MTN MoMo + Airtel Money client
│           ├── africastalking/    SMS + USSD services
│           ├── fhir/              FHIR R4 resource builders (SNOMED CT)
│           └── dicom/             DICOM parsing + fundus extraction
├── mobile/
│   └── retinalai/           Flutter app (Drift DB, Riverpod, ONNX Runtime, camera)
│       ├── lib/services/    Inference, gate, audit, sync, connectivity
│       ├── lib/screens/     Splash, home, camera, screening, sync, settings
│       └── lib/data/        Drift database tables + DAO
├── frontend/                Next.js 16 (Zustand, TanStack Query, Tailwind)
├── src/
│   ├── models/              RetinalFoundationHybrid + 4 legacy GNN architectures
│   ├── agents/              LangGraph 6-node pipeline + 3 autonomous agents + event bus
│   ├── serving/             Ray Serve client, circuit breaker, canary router, edge runtime
│   ├── active_learning/     Uncertainty-based flagging + human review loop
│   ├── data/                Dataset, augmentation, datamodule, fundus gate v2
│   ├── training/            DDP trainer, losses, metrics, HPO
│   ├── monitoring/          Data/model drift detection, health SLA tracking
│   ├── governance/          Bias auditor, audit logs, model cards, fairness evaluator
│   ├── optimization/        Quantization (INT8/FP16) + export (ONNX/TorchScript/CoreML/TRT)
│   └── visualization/       IEEE publication plots
├── tests/                   210+ tests (API, models, AL, monitoring, bias, gate, offline, mobile)
├── configs/
│   ├── train.yaml           Training config (8x RTX A6000, ViGNN)
│   ├── hybrid_2026.yaml     RETFound + LoRA + MoE config
│   ├── backend_2026.yaml    All 2026 env vars reference
│   ├── otel-collector-config.yaml  OTEL Collector pipeline
│   └── prometheus.yml       Prometheus scrape config
├── k8s/
│   ├── base/                Backend deployment + HPA + PDB + ServiceAccount
│   └── chaos/               LitmusChaos experiments (pod-delete, network, latency)
├── scripts/                 Training, export, SBOM, benchmark, bias audit
├── docs/                    23 documentation files + architecture
├── docker-compose.yml       Base (API + API-CPU + HF)
├── docker-compose.otel.yml  Phase 1: OTEL Collector + Jaeger + Prometheus
├── docker-compose.mlflow.yml  Phase 1: MLflow tracking server
├── docker-compose.2026.yml  Full stack (all phases)
├── Dockerfile               GPU backend (nvidia/cuda:12.1.1)
├── Dockerfile.cpu           CPU-only backend
├── Dockerfile.hf            HF Spaces (python:3.11-slim, CPU PyTorch, supervisord)
├── supervisord.conf         Process manager (backend + frontend + nginx)
├── nginx.conf               Reverse proxy (port 7860 → backend:8080 + frontend:3000)
├── train.py                 DDP training entry point
├── dvc.yaml                 Reproducible ML pipeline
└── pyproject.toml           UV dependencies (9 optional groups)
```

## Documentation

| Doc | Topic |
|---|---|
| [Data Ingestion](docs/01-data-ingestion.md) | Dataset download, format, class imbalance |
| [Data Augmentation](docs/02-data-augmentation.md) | Medical imaging augmentation pipeline |
| [Training](docs/03-training.md) | Multi-GPU DDP training, loss functions, models |
| [Evaluation](docs/04-evaluation.md) | Metrics, benchmarks, IEEE plots |
| [Model Export](docs/05-model-export.md) | Checkpoint to production deployment |
| [Backend Setup](docs/06-backend-setup.md) | FastAPI API server, auth, middleware |
| [Frontend Setup](docs/07-frontend-setup.md) | Next.js 16 + Zustand + TanStack Query |
| [Production Improvements](docs/08-production-improvements.md) | Monitoring, drift detection, health SLA |
| [Testing](docs/09-testing.md) | Test suite, data validation, CI quality gates |
| [Security](docs/10-security.md) | Auth, rate limiting, scanning, SBOM |
| [Governance & Compliance](docs/11-governance.md) | Model cards, fairness, audit trail, EU AI Act |
| [Advanced MLOps](docs/12-advanced-mlops.md) | DVC pipelines, HPO, retraining, orchestration |
| [Commercialization](docs/13-commercialization-strategy.md) | GTM strategy, pricing, regulatory pathway |
| [Image Gating](docs/15-image-gating.md) | Pre-inference fundus validation (3-layer + v2 fusion) |
| [Fundus Gate V2](docs/16-fundus-gate-v2.md) | Safety architecture: learned gate, fusion, rollout plan |
| [Implementation Roadmap](docs/17-implementation-roadmap.md) | 2026 transformation: timeline, risk matrix, architecture diagram |
| [Migration Guide](docs/18-migration-guide.md) | Phase-by-phase activation commands + rollback strategies |
| [Offline Mobile](docs/19-offline-mobile-optimization.md) | Offline-first architecture, delta sync, mobile bundle |
| [Deployment Rollout](docs/20-deployment-rollout-guide.md) | Deployment guide for Uganda clinics |
| [vLLM GPU 7 AWQ Runbook](docs/21-vllm-gpu7-awq-optimization.md) | Qwen3-8B-AWQ runtime profile, verification, rollback |
| [Crane Cloud Deployment](docs/22-crane-cloud-deployment.md) | Production deployment on Crane Cloud K8s (Uganda) |
| [Billing Platform](docs/23-billing-platform.md) | Subscription, auth, quota, all payment rails, email, renewal cron, seats |
| [ISO 14971 Risk Analysis](docs/iso14971_risk_analysis.md) | Medical device risk management (12 hazards, controls) |
| [Architecture](docs/architecture-retinal-foundation-hybrid.md) | RetinalFoundationHybrid V1+V2 architecture deep-dive |
| [Bill of Materials](docs/Bill-of-Materials.md) | Hardware + software BOM with Uganda Shilling pricing |
| [IEEE Final Report](docs/IEEE_Final_Report.md) | Academic paper: offline-first retinal detection for rural Uganda |

## Models

### Production Model (2026)

| Model | Total Params | Trainable (LoRA) | p99 Latency (A100 INT8) | Innovation |
|---|---|---|---|---|
| **RetinalFoundationHybridV2** | 305M | 2.4M (LoRA r16) | <4ms | RETFound ViT-L + LoRA + Graph + Bottleneck + ASL |
| **MobileStudentV1** | 5.2M | 5.2M | <1.8s (mobile) | MobileNetV3-Large distilled from HybridV2, INT8 ONNX |

### Legacy Models (deprecated, available for comparison)

| Model | Params | Latency | Innovation |
|---|---|---|---|
| ViGNN | 26.1M | 25ms | Graph message passing + disease prototypes |
| GraphCLIP | 24.8M | 28ms | Dynamic graph adjacency + sparse attention |
| VisualLanguageGNN | 24.3M | 22ms | Cross-modal visual-text fusion |
| SceneGraphTransformer | 31.2M | 26ms | Ensemble branches + uncertainty calibration |

## API Endpoints

### Core

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root info (app, version, GPU, model status) |
| `GET` | `/health` | Health check (status, device, diseases count) |
| `GET` | `/health/model` | Detailed model health (latency p50/p95/p99, SLA) |
| `GET` | `/health/gate` | Fundus gate v2 metrics (pass rate, latency, alerts) |
| `POST` | `/api/v1/predict` | Image prediction with clinical reasoning |
| `GET` | `/api/v1/diseases` | List of 45 detectable diseases |
| `POST` | `/api/v1/auth/token` | JWT access token exchange |

### Fundus Gate

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/gate/status` | Gate config, stats, learned model status |
| `POST` | `/api/v1/gate/validate` | Debug: full gate breakdown (always 200) |

### Human Review

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/review/pending` | Pending human review requests |
| `POST` | `/api/v1/review/{id}/resolve` | Resolve a review decision (triggers active learning) |
| `GET` | `/api/v1/review/stats` | Review queue statistics |

### Explainability

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/explain/gradcam` | GradCAM heatmap for target diseases |
| `POST` | `/api/v1/explain/lime` | LIME superpixel importance |
| `POST` | `/api/v1/explain/shap` | SHAP feature importance |
| `POST` | `/api/v1/explain/integrated-gradients` | Attribution-based explanation |
| `POST` | `/api/v1/explain/eli5` | Human-readable explanation |
| `POST` | `/api/v1/explain/comprehensive` | All methods combined |
| `GET` | `/api/v1/explain/available` | Available explainability methods |

### Clinical & Agents

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/clinical/disease-info/{code}` | Disease information from knowledge graph |
| `POST` | `/api/v1/clinical/explain-reasoning` | KG-based clinical reasoning |
| `POST` | `/api/v1/agents/screen` | Full LangGraph agentic screening pipeline |
| `GET` | `/api/v1/agents/status` | Agent orchestrator status |

### Governance (Phase 1+3)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/governance/drift` | Drift detection status + history |
| `GET` | `/api/v1/governance/active-learning-stats` | Active learning queue + fine-tune history |
| `GET` | `/api/v1/governance/model-registry` | MLflow registry status |
| `GET` | `/api/v1/governance/fairness` | Fairness dashboard with demographic breakdowns |
| `GET` | `/api/v1/governance/model-card` | Current model card (JSON or Markdown) |
| `GET` | `/api/v1/governance/audit` | Query immutable audit log |
| `GET` | `/api/v1/governance/audit/integrity` | Verify audit chain integrity (SHA-256) |

### Voice (Phase 5)

| Method | Path | Description |
|---|---|---|
| `WS` | `/v1/voice/stream` | WebSocket for streaming ASR/TTS with Luganda support |

### Uganda Health Ecosystem (Phase 5)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/dhis2/patient/search` | Search patients in DHIS2 by name or NIN |
| `POST` | `/api/v1/dhis2/referral` | Create referral event (offline queue fallback) |
| `POST` | `/api/v1/payments/request` | Initiate MTN MoMo or Airtel Money payment |
| `POST` | `/api/v1/sms/send-referral` | Send bilingual referral SMS |
| `POST` | `/api/v1/sms/ussd` | USSD session callback for feature phones |
| `GET` | `/api/v1/fhir/DiagnosticReport/{id}` | FHIR R4 DiagnosticReport with SNOMED CT codes |
| `GET` | `/api/v1/fhir/Bundle/{id}` | FHIR R4 Bundle (report + observations) |
| `POST` | `/api/v1/dicom/upload` | Upload DICOM file, extract fundus images |
| `POST` | `/api/v1/offline/bundle/delta` | Delta sync for mobile bundle updates |

### Edge Inference (Phase 3)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict/onnx` | ONNX Runtime inference |
| `POST` | `/api/v1/predict/coreml` | Core ML inference (Apple Silicon) |
| `POST` | `/api/v1/predict/quantized` | INT8/FP16 quantized inference |
| `GET` | `/api/v1/predict/edge/status` | Loaded edge formats + config |

## Make Targets

```bash
# Development
make install          # Install backend (uv) + frontend (bun)
make install-backend  # Install backend only (uv sync)
make install-frontend # Install frontend only (bun install)
make dev              # Run backend + frontend in parallel
make backend          # Run backend only (uvicorn, port 8080)
make frontend         # Run frontend only (bun dev, port 3000)
make build-frontend   # Build frontend for production
make test             # Run 210+ tests
make test-fast        # Run tests (fail-fast mode)

# Training
make train            # 8-GPU DDP training
make train-4gpu       # 4-GPU DDP training
make train-1gpu       # Single GPU training (CUDA:2)
make validate-data    # Data quality validation
make hpo              # Optuna hyperparameter optimization
make pipeline         # Full pipeline: train + plots

# Production
make export           # ONNX + TorchScript export
make export-all       # Export all formats with parity validation
make model-card       # Generate model & dataset cards
make sbom             # Generate SBOM (Syft + Grype)
make mlops-pipeline   # Full: validate -> train -> export -> model card

# 2026 Infrastructure
make up-phase1        # OTEL + Jaeger + Prometheus + MLflow
make up-phase2        # Phase 1 + Ray Serve + Kafka
make up-full          # Full 2026 stack (all phases)
make down-full        # Teardown full stack

# Mobile/Offline
make distill          # Distill teacher to MobileNetV3-Large student (GPU required)
make export-mobile    # Export student + gate ONNX INT8 + bundle
make pilot-readiness  # Validate national pilot readiness (12 checks)

# Governance
make bias-audit-uganda  # Uganda-specific bias audit (F1 disparity < 0.08)
make federated-sim      # Simulate federated learning (5 clients, Dirichlet split)
make moh-package        # Generate Uganda MoH regulatory submission

# Fundus Gate V2
make test-gate        # Run 57 gate tests (24 unit + 33 adversarial)
make benchmark-gate   # Gate latency benchmarks (p50/p95/p99)

# Deployment
make deploy-hf        # Deploy to Hugging Face Spaces
make hf-login         # Authenticate with HuggingFace CLI
make hf-local         # Local test of HF Spaces Docker image

# Utilities
make check-retrain    # Check if retraining is needed
make dvc-repro        # Reproduce DVC pipeline
make plots            # Generate IEEE publication plots
make plots-eda        # Generate EDA plots only
make clean            # Remove checkpoints, cache
```

## 2026 Production Features

All features are **opt-in via environment variables** (`FEATURE__ENABLED=true`). Disabled by default for backward compatibility.

| Phase | Feature | Env Var | Description |
|-------|---------|---------|-------------|
| 1 | OpenTelemetry | `TELEMETRY__ENABLED` | Distributed tracing + metrics (Jaeger + Prometheus) |
| 1 | MLflow Registry | `MLFLOW__ENABLED` | Model versioning, staging/production promotion |
| 1 | Active Learning | `ACTIVE_LEARNING_LOOP__ENABLED` | Review -> LoRA fine-tune -> register closed loop |
| 1 | Drift Detection | `DRIFT__ENABLED` | PSI + KS + NannyML + Evidently + webhook alerts |
| 2 | Ray Serve | `RAY__ENABLED` | Dynamic batching, autoscaling, canary releases |
| 2 | Kafka Audit | `KAFKA__ENABLED` | Immutable audit logs (Kafka -> Iceberg) |
| 2 | mTLS | `MTLS__ENABLED` | Mutual TLS between services |
| 3 | Edge ONNX | `EDGE__ONNX_ENABLED` | ONNX Runtime inference endpoint |
| 3 | Fairness | `FAIRNESS__ENABLED` | Demographic fairness dashboard |
| 3 | Model Cards | `MODEL_CARD__AUTO_GENERATE` | Auto-generate on MLflow promotion |
| 4 | Resilience | `RESILIENCE__ENABLED` | Graceful degradation + health-aware routing |
| 4 | Multi-Modal | `MULTIMODAL__ENABLED` | Fundus + OCT + metadata fusion (skeleton) |
| 4 | Federated | `FEDERATED__ENABLED` | Flower LoRA-only exchange + secure aggregation |
| 5 | Offline RAG | `OFFLINE_RAG__ENABLED` | FAISS + ONNX embedder + delta sync bundles |
| 5 | Quantization | `QUANTIZATION__ENABLED` | GGUF/AWQ/GPTQ/INT8 with quality gates |
| 5 | Voice-First | `VOICE_FIRST__ENABLED` | Whisper ASR + Piper TTS + Silero VAD |
| 5 | DHIS2 | `DHIS2__ENABLED` | Uganda health information system integration |
| 5 | Mobile Money | `MOBILE_MONEY__ENABLED` | MTN MoMo + Airtel Money referral payments |
| 5 | SMS/USSD | `AFRICASTALKING__ENABLED` | Africa's Talking for feature phone fallback |

See [`configs/backend_2026.yaml`](configs/backend_2026.yaml) for the complete configuration reference.

## Tech Stack

**ML/Training**: PyTorch 2.6+ / CUDA 12, timm, torchmetrics, W&B, Optuna, LoRA
**Distillation**: MobileNetV3-Large student, precision-aware KD loss, temperature annealing
**Backend**: FastAPI 3.0, UV, Pydantic Settings (nested), JWT auth, structured JSON logging
**Frontend**: Next.js 16, Bun, Zustand 5, TanStack Query 5, Tailwind CSS
**Mobile**: Flutter 3.24+, Drift (reactive SQLite), Riverpod, ONNX Runtime Mobile
**Voice**: Whisper-tiny (faster-whisper), Piper TTS, Silero VAD, Luganda + English
**Uganda**: DHIS2, MTN MoMo, Airtel Money, Africa's Talking SMS/USSD, PDP Act 2019
**Clinical**: FHIR R4 (SNOMED CT), DICOM (pydicom), bilingual referral letters
**Observability**: OpenTelemetry SDK, Jaeger, Prometheus, OTEL Collector
**ML Registry**: MLflow 3.0 (staging/production/shadow deployment)
**Serving**: Ray Serve (dynamic batching, canary releases, circuit breakers), vLLM OpenAI API (`Qwen/Qwen3-8B-AWQ`, AWQ INT4, GPU 7 right-sized to ~15.5 GB)
**Edge**: ONNX Runtime, Core ML, INT8/FP16 quantization, offline bundle + delta sync
**Security**: mTLS, JWT, SBOM (Syft + Grype), rate limiting, security headers
**Audit**: Kafka + Apache Iceberg (immutable) with JSONL fallback, SHA-256 hash chain
**Governance**: Model cards, fairness dashboard, Uganda bias audit, EU AI Act, ISO 14971
**Agents**: LangGraph 7-node pipeline (with history extraction), Claude + Groq + deterministic
**Federated**: Flower LoRA-only exchange, Opacus DP-SGD, secure aggregation
**Resilience**: Circuit breakers, graceful degradation (4 levels), LitmusChaos
**Infrastructure**: Docker Compose (4 configs), Kubernetes (HPA + PDB + chaos), DVC
**Testing**: pytest (200+ tests), data validation, CI quality gates
**CI/CD**: GitHub Actions (lint, test, bias audit, bundle size, faithfulness gates)

## License

CC-BY-4.0

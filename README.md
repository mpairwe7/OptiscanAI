# RetinalAI Clinical Screening Platform v3.0

Multi-label retinal disease classification (24 diseases) powered by **RETFound ViT-Large** foundation model with clinical knowledge graph reasoning, LangGraph agentic workflow, and a full 2026 production MLOps stack. Trained on 8x NVIDIA RTX A6000 GPUs. **198 tests passing**, distributed tracing, model registry, active learning, and EU AI Act compliance.

**Precision Rescue Results (v2):** Precision 0.312 (12.5x over v1), F1 0.362 (7.9x), AUC 0.888 (+85%), Accuracy 95.4%.

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
  Serving:     Ray Serve (dynamic batching, canary)      (Phase 2)
  Security:    mTLS + SBOM + Kafka+Iceberg audit         (Phase 2)
  Resilience:  Circuit breakers + chaos engineering       (Phase 2)
  Governance:  Auto model cards + fairness dashboard     (Phase 3)
  Edge:        ONNX + CoreML + INT8 inference endpoints  (Phase 3)
  Future:      Multi-modal fusion + federated learning   (Phase 4)
  Safety:      Fundus gate v2 (statistical + learned MobileNetV3 fusion, <12ms p99)
  Agents:      LangGraph 6-node pipeline (Claude + Groq fallback)
  CI/CD:       GitHub Actions (lint, test, security scan, Docker, deploy)
```

## Quick Start

```bash
# Install (core)
uv sync && cd frontend && bun install && cd ..

# Install with production features (Phase 1-3)
pip install -e ".[observability,drift-detection,ray-serve,edge]"

# Development (backend:8080 + frontend:3000)
make dev

# Run tests (188 tests)
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

## Project Structure

```
.
├── backend/
│   └── app/
│       ├── core/
│       │   ├── config.py              Nested Pydantic settings (16 feature sections)
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
│       │   └── logging_config.py      Structured JSON logging
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
│       │   └── auth.py                /api/v1/auth/token
│       └── serving/
│           └── ray_serve_config.py    Ray Serve deployment (Phase 2)
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
├── tests/                   188 tests (API, models, active learning, monitoring, bias, gate)
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
├── docs/                    18 documentation files + architecture
├── docker-compose.yml       Base (API + API-CPU + HF)
├── docker-compose.otel.yml  Phase 1: OTEL Collector + Jaeger + Prometheus
├── docker-compose.mlflow.yml  Phase 1: MLflow tracking server
├── docker-compose.2026.yml  Full stack (all phases)
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
| **[Implementation Roadmap](docs/17-implementation-roadmap.md)** | **2026 transformation: timeline, risk matrix, architecture diagram** |
| **[Migration Guide](docs/18-migration-guide.md)** | **Phase-by-phase activation commands + rollback strategies** |

## Models

### Production Model (2026)

| Model | Total Params | Trainable (LoRA) | p99 Latency (A100 INT8) | Innovation |
|---|---|---|---|---|
| **RetinalFoundationHybrid** | 315M | 11.1M | <4ms | RETFound ViT-L + LoRA + Graph Head + MoE + UQ |

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
make dev              # Run backend + frontend in parallel
make test             # Run 188 tests
make test-fast        # Run tests (fail-fast mode)

# Training
make train            # 8-GPU DDP training
make validate-data    # Data quality validation
make hpo              # Optuna hyperparameter optimization

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

# Utilities
make check-retrain    # Check if retraining is needed
make dvc-repro        # Reproduce DVC pipeline
make plots            # Generate IEEE publication plots
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
| 4 | Federated | `FEDERATED__ENABLED` | Flower / NVFlare client (skeleton) |

See [`configs/backend_2026.yaml`](configs/backend_2026.yaml) for the complete configuration reference.

## Tech Stack

**ML/Training**: PyTorch 2.6+ / CUDA 12, timm, torchmetrics, W&B, Optuna, LoRA
**Backend**: FastAPI 3.0, UV, Pydantic Settings (nested), JWT auth, structured JSON logging
**Frontend**: Next.js 16, Bun, Zustand 5, TanStack Query 5, Tailwind CSS
**Observability**: OpenTelemetry SDK, Jaeger, Prometheus, OTEL Collector
**ML Registry**: MLflow 3.0 (staging/production/shadow deployment)
**Serving**: Ray Serve (dynamic batching, canary releases, circuit breakers)
**Edge**: ONNX Runtime, Core ML, INT8/FP16 quantization
**Security**: mTLS, JWT, SBOM (Syft + Grype), rate limiting, security headers
**Audit**: Kafka + Apache Iceberg (immutable, queryable) with JSONL fallback
**Governance**: Auto model cards, fairness dashboard, bias auditing, EU AI Act
**Agents**: LangGraph 6-node pipeline, Claude (primary) + Groq (fallback) + deterministic
**Resilience**: Circuit breakers, graceful degradation (4 levels), LitmusChaos
**Infrastructure**: Docker Compose (4 configs), Kubernetes (HPA + PDB + chaos), DVC
**Testing**: pytest (188 tests), data validation, CI quality gates
**CI/CD**: GitHub Actions (lint, test, security scan, Docker publish, deploy)

## License

CC-BY-4.0

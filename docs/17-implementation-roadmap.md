# RetinalAI 2026 Production Transformation — Implementation Roadmap

## Executive Summary

Transform the RetinalAI Clinical Screening Platform from a strong single-instance FastAPI application into a world-class 2026 medical AI production system across 4 phases, adding distributed tracing, model registry, active learning, scalable serving, zero-trust security, governance automation, and edge deployment.

## Phased Timeline

| Phase | Focus | Duration | Prerequisites |
|-------|-------|----------|---------------|
| **Phase 1** | Observability, MLOps & Active Learning | 5–6 weeks | None |
| **Phase 2** | Scalability, Security & Resilience | 6–7 weeks | Phase 1 |
| **Phase 3** | Governance, Fairness & Edge | 4–5 weeks | Phase 1 (Phase 2 optional) |
| **Phase 4** | Future-Proofing | 3–4 weeks | Phases 1–2 |

**Total: 18–22 weeks (4.5–5.5 months)**

## Risk Matrix

| Risk | Phase | Likelihood | Impact | Score | Mitigation |
|------|-------|-----------|--------|-------|------------|
| OTEL overhead degrades p99 latency | 1 | Medium | Medium | 6 | Configurable sampling rate; start at 10% in production |
| MLflow downtime blocks model loading | 1 | Low | High | 6 | Fallback to local model path; ModelService.load() handles missing files |
| Active learning retraining degrades quality | 1 | Medium | High | 9 | Evaluate on holdout test set before MLflow promotion |
| Ray Serve cold start latency spikes | 2 | High | Medium | 8 | Warm replicas; pre-load model on worker startup |
| Kafka message loss during high throughput | 2 | Low | High | 6 | acks=all, replication factor 3, JSONL dual-write |
| mTLS certificate rotation outage | 2 | Medium | High | 9 | Automated rotation with 30-day lead time |
| ONNX export loses accuracy vs PyTorch | 3 | Medium | High | 9 | Automated parity test (max diff < 1e-4) gates export |
| Fairness metrics reveal disparities | 3 | Medium | High | 9 | Have remediation plan ready; this is a finding, not failure |
| Federated aggregation diverges | 4 | Medium | High | 9 | Skeleton only; no production federation until validated |

## New System Capabilities After Full Implementation

| Capability | Before | After |
|-----------|--------|-------|
| Observability | Custom JSON logs + RequestID | OpenTelemetry distributed tracing + Jaeger + Prometheus |
| Model Registry | File-based checkpoints | MLflow 3.0 with staging/production/shadow deployment |
| Active Learning | Manual review queue | Closed loop: review → LoRA fine-tune → MLflow registration |
| Drift Detection | In-memory PSI/KS | Enhanced with NannyML + Evidently + webhook alerts |
| Model Serving | Single FastAPI process | Ray Serve with dynamic batching + canary releases |
| Security | JWT + rate limiting | mTLS + SBOM + immutable audit logs (Kafka + Iceberg) |
| Audit Trail | JSONL files | Kafka → Apache Iceberg (immutable, queryable) |
| Governance | Manual model cards | Auto-generated model cards + fairness dashboard |
| Edge Deployment | Export scripts only | Live ONNX/CoreML/INT8 inference endpoints |
| Resilience | LLM fallback chain | Circuit breakers + chaos engineering + graceful degradation |
| Multi-Modal | Fundus only | Skeleton for Fundus + OCT + metadata fusion |
| Federated | None | Flower/NVFlare client interface ready |

## Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Frontend<br/>Next.js 16]
        MOBILE[Mobile/Edge]
        EHR[EHR Integration]
    end

    subgraph "API Gateway"
        NGINX[NGINX + mTLS]
    end

    subgraph "FastAPI Application"
        OTEL_MW[OTEL Middleware]
        REQID[RequestID Middleware]
        RATE[Rate Limiter]
        SEC[Security Headers]

        subgraph "Routers"
            PREDICT[/predict]
            PREDICT_EDGE[/predict/onnx,coreml,quantized]
            REVIEW[/review]
            GOVERN[/governance]
            AGENTS_R[/agents]
        end

        subgraph "Core Services"
            MS[ModelService<br/>Strategy Pattern]
            AL[ActiveLearningLoop]
            DRIFT[EnhancedDriftDetector]
        end

        subgraph "Inference Backends"
            LOCAL[LocalBackend<br/>PyTorch]
            RAY_B[RayServeBackend]
            EDGE_RT[EdgeRuntime<br/>ONNX/CoreML/INT8]
        end
    end

    subgraph "Agent Orchestrator"
        SA[ScreeningAgent<br/>LangGraph 6-node]
        MA[MonitorAgent]
        GA[GovernanceAgent]
        EB[EventBus<br/>In-process + Kafka]
    end

    subgraph "LLM Layer"
        CB_LLM[Circuit Breaker]
        CLAUDE[Claude API]
        GROQ[Groq API]
        DETERM[Deterministic<br/>Fallback]
    end

    subgraph "Observability"
        OTEL_C[OTEL Collector]
        JAEGER[Jaeger<br/>Traces]
        PROM[Prometheus<br/>Metrics]
    end

    subgraph "ML Infrastructure"
        MLFLOW[MLflow<br/>Registry]
        RAY_S[Ray Serve<br/>Cluster]
    end

    subgraph "Data Infrastructure"
        KAFKA[Kafka]
        ICE[Iceberg<br/>Audit Tables]
    end

    subgraph "Governance"
        FAIR[Fairness<br/>Dashboard]
        MCARD[Model Card<br/>Generator]
        KG[Clinical<br/>Knowledge Graph]
    end

    WEB --> NGINX
    MOBILE --> NGINX
    EHR --> NGINX
    NGINX --> OTEL_MW --> REQID --> RATE --> SEC

    SEC --> PREDICT
    SEC --> PREDICT_EDGE
    SEC --> REVIEW
    SEC --> GOVERN
    SEC --> AGENTS_R

    PREDICT --> MS
    PREDICT_EDGE --> EDGE_RT
    MS --> LOCAL
    MS --> RAY_B
    RAY_B --> RAY_S

    REVIEW --> AL
    AL --> MLFLOW

    MS --> DRIFT
    DRIFT --> EB

    SA --> CB_LLM
    CB_LLM --> CLAUDE
    CB_LLM --> GROQ
    CB_LLM --> DETERM

    OTEL_MW --> OTEL_C
    OTEL_C --> JAEGER
    OTEL_C --> PROM

    EB --> KAFKA
    KAFKA --> ICE

    GOVERN --> FAIR
    GOVERN --> MCARD
    MS --> KG
```

## Phase Dependencies

```
Phase 1 (Observability + MLOps)
    ├── Phase 2 (Scalability + Security) ── requires OTEL + MLflow
    │       └── Phase 4 (Future-Proofing) ── requires circuit breakers
    └── Phase 3 (Governance + Edge) ── requires MLflow for model cards
```

## Component Details

### Phase 1 Components

**OpenTelemetry (`backend/app/core/telemetry.py`)**
- Auto-instrumentation for all FastAPI HTTP requests via `FastAPIInstrumentor`
- Custom spans: `retinalai.model.inference`, `retinalai.kg.clinical_reasoning`, `retinalai.llm.claude`, `retinalai.llm.groq`
- 8 metric instruments: prediction count, inference duration, model confidence, diseases detected, AL queue size, drift checks, review count, LLM call duration
- `@traced` decorator for wrapping any sync/async function in a span
- No-op fallbacks when OTEL SDK is not installed (zero overhead)
- Request ID bridged from existing `RequestIDMiddleware` into OTEL span context

**MLflow Registry (`backend/app/core/mlflow_registry.py`)**
- Model registration with full lineage (training config, dataset hash, metrics)
- Staging -> Production promotion with configurable validation gates (min F1, min AUC)
- Shadow deployment tracking: run staging + production side-by-side, compare agreement
- A/B test metadata logging
- Event bus integration: emits `MODEL_PROMOTED` on successful promotion

**Active Learning Loop (`backend/app/core/active_learning.py`)**
- Hooks into `POST /api/v1/review/{id}/resolve` — when decision is "modified" with corrected labels, sample is queued
- Persists corrected samples to `data/active_learning/corrected/` as JSON
- Triggers LoRA fine-tuning when queue reaches threshold (default: 150 samples)
- Mixes corrected samples with high-confidence predictions for knowledge retention
- Fine-tuned model registered in MLflow, events emitted for lifecycle tracking

**Enhanced Drift Detection (`backend/app/core/drift_detector.py`)**
- Wraps existing PSI + KS-test detectors from `src/monitoring/drift.py`
- Optional NannyML CBPE for estimated performance monitoring (no ground truth needed)
- Optional Evidently DataDriftPreset for multivariate drift
- Webhook alerts when drift exceeds thresholds
- Auto-check every N predictions (configurable interval)

### Phase 2 Components

**Ray Serve (`backend/app/serving/ray_serve_config.py`)**
- `@serve.batch` dynamic batching: collects up to 16 images over 100ms window
- Autoscaling: 1-8 replicas based on `target_ongoing_requests`
- ModelService abstraction: `LocalBackend` (current) vs `RayServeBackend` (HTTP to Ray)

**Circuit Breakers (`src/serving/circuit_breaker.py`)**
- Three states: CLOSED (normal), OPEN (rejecting), HALF_OPEN (testing recovery)
- Configurable failure threshold, recovery timeout, half-open call limit
- `CircuitBreakerRegistry` for managing multiple named breakers
- Event bus integration: emits `CIRCUIT_BREAKER_OPENED` / `CLOSED`

**Canary Router (`src/serving/canary_router.py`)**
- Weighted traffic routing between model versions (0-100% canary weight)
- CRC-32 consistent hashing for sticky sessions
- Per-version call count tracking and actual traffic split metrics

**Immutable Audit (`backend/app/core/audit_logger.py`)**
- SHA-256 chained entries (blockchain-like tamper detection)
- Kafka producer (confluent-kafka) for durable event streaming
- JSONL fallback when Kafka is unavailable
- Monthly file rotation with queryable history

### Phase 3 Components

**Edge Runtime (`src/serving/edge_runtime.py`)**
- Unified loader for ONNX, CoreML, and quantized PyTorch models
- Output parity validation against FP32 reference (configurable tolerance)
- Same preprocessing pipeline as ModelService (224x224, ImageNet normalize)

**Model Card Generator (`backend/app/core/model_card_generator.py`)**
- Auto-generates on `MODEL_PROMOTED` event from MLflow
- 7 sections: model details, intended use, training data, evaluation, ethics, limitations, regulatory
- Outputs both JSON and Markdown formats

### Phase 4 Components

**Graceful Degradation (`backend/app/core/graceful_degradation.py`)**
- 4 levels: FULL -> AGENT_DEGRADED -> RULES_ONLY -> MODEL_ONLY
- Health-aware routing adjusts pipeline depth based on service availability
- Integrates with circuit breakers for automatic detection
- Full pipeline invokes LangGraph agent via `run_screening()`

**Multi-Modal Fusion (`backend/app/core/multi_modal_fusion.py`)**
- Abstract `ModalityEncoder` and `FusionStrategy` interfaces
- Concrete encoders: FundusEncoder (wraps ViGNN), OCTEncoder (ViT-Small), PatientMetadataEncoder (MLP)
- Fusion strategies: concatenation + projection, cross-attention
- Missing modalities handled via learned default embeddings

**Federated Client (`backend/app/core/federated_client.py`)**
- Abstract `FederatedClient` with fit/evaluate/get_params/set_params
- `FlowerRetinalClient`: full training loop with AdamW, gradient clipping, optional DP noise
- `FedAvgStrategy` / `FedProxStrategy` for aggregation

## Production Readiness Checklist (2026 Standards)

### Observability
- [ ] OpenTelemetry traces for all HTTP requests (auto-instrumentation)
- [ ] Custom spans: model inference, KG reasoning, LLM calls, explainability
- [ ] Metrics: prediction count, inference latency, drift score, AL queue size
- [ ] Jaeger UI accessible for trace exploration
- [ ] Prometheus metrics scraped and alertable

### MLOps
- [ ] MLflow model registry with staging → production promotion
- [ ] Shadow deployment comparison for model versions
- [ ] Active learning closed loop: review → fine-tune → register → promote
- [ ] Drift detection: PSI, KS-test, confidence drop + NannyML + Evidently
- [ ] Automated alerts on drift threshold breach

### Security
- [ ] mTLS between all internal services
- [ ] SBOM generated on every Docker build (Syft + Grype)
- [ ] Immutable audit logs (Kafka + Apache Iceberg)
- [ ] JWT authentication with RBAC
- [ ] Security headers (HSTS, CSP, X-Frame-Options)
- [ ] No secrets in code (all via environment variables)

### Scalability
- [ ] Ray Serve with dynamic batching (16 images, 100ms window)
- [ ] Autoscaling: 1–8 replicas based on request load
- [ ] Canary releases with configurable traffic split
- [ ] Circuit breakers on all external service calls

### Governance
- [ ] Auto-generated model cards after every promotion
- [ ] Fairness dashboard with demographic breakdowns
- [ ] Audit log chain integrity verification (SHA-256)
- [ ] Regulatory mode support: research / CE-marked / FDA-cleared

### Edge
- [ ] ONNX Runtime inference endpoint
- [ ] Core ML inference endpoint (Apple Silicon)
- [ ] INT8/FP16 quantized inference endpoint
- [ ] Output parity validation (±1e-4 vs PyTorch)

### Resilience
- [ ] Circuit breakers: Claude, Groq, Ray Serve, Kafka, MLflow
- [ ] Graceful degradation: FULL → AGENT_DEGRADED → RULES_ONLY → MODEL_ONLY
- [ ] Chaos engineering experiments (LitmusChaos)
- [ ] Health-aware routing with automatic fallback

### Compliance
- [ ] EU AI Act Article 12: Audit trail with lifecycle logging
- [ ] FDA SaMD: Configuration management via MLflow + DVC
- [ ] Data governance: Dataset cards, bias auditing
- [ ] Human oversight: Review system with clinical escalation

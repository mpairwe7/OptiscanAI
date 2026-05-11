# RetinalAI Migration & Deployment Guide

Step-by-step migration from the current backend to the full 2026 production system.

## Prerequisites

- Docker + Docker Compose v2
- Python 3.10+ with UV package manager
- NVIDIA GPU with CUDA 12.1+ (for GPU features)
- kubectl + helm (for Kubernetes deployment)

## Phase 1: Observability, MLOps & Active Learning

### Step 1.1 — Install Dependencies

```bash
# Install Phase 1 optional dependencies
pip install -e ".[observability]"

# Or with UV
uv sync --extra observability
```

### Step 1.2 — Enable OpenTelemetry

```bash
# Start observability stack
docker compose -f docker-compose.yml -f docker-compose.otel.yml up -d

# Verify services
curl http://localhost:16686  # Jaeger UI
curl http://localhost:9090   # Prometheus UI

# Enable in API
export TELEMETRY__ENABLED=true
export TELEMETRY__OTLP_ENDPOINT=http://localhost:4317
```

### Step 1.3 — Enable MLflow Registry

```bash
# Start MLflow
docker compose -f docker-compose.yml -f docker-compose.mlflow.yml up -d

# Verify MLflow UI
curl http://localhost:5000

# Enable in API
export MLFLOW__ENABLED=true
export MLFLOW__TRACKING_URI=http://localhost:5000
```

### Step 1.4 — Enable Active Learning Loop

```bash
# Create data directory
mkdir -p data/active_learning/corrected data/active_learning/processed

# Enable
export ACTIVE_LEARNING_LOOP__ENABLED=true
export ACTIVE_LEARNING_LOOP__RETRAIN_THRESHOLD=150

# Verify stats endpoint
curl http://localhost:8080/api/v1/governance/active-learning-stats
```

### Step 1.5 — Enable Enhanced Drift Detection

```bash
# Already enabled by default (DRIFT__ENABLED=true)
# Optional: enable NannyML/Evidently
pip install nannyml evidently
export DRIFT__NANNYML_ENABLED=true
export DRIFT__EVIDENTLY_ENABLED=true

# Verify drift endpoint
curl http://localhost:8080/api/v1/governance/drift
```

### Phase 1 Verification

```bash
# Check all Phase 1 endpoints
curl http://localhost:8080/api/v1/governance/drift
curl http://localhost:8080/api/v1/governance/active-learning-stats
curl http://localhost:8080/api/v1/governance/model-registry

# Check traces in Jaeger
open http://localhost:16686

# Check metrics in Prometheus
open http://localhost:9090
```

### Phase 1 Rollback

```bash
# Disable all Phase 1 features (zero code changes needed)
export TELEMETRY__ENABLED=false
export MLFLOW__ENABLED=false
export ACTIVE_LEARNING_LOOP__ENABLED=false

# Stop infrastructure
docker compose -f docker-compose.yml -f docker-compose.otel.yml -f docker-compose.mlflow.yml down
```

---

## Phase 2: Scalability, Security & Resilience

### Step 2.1 — Install Dependencies

```bash
pip install -e ".[ray-serve,kafka,security]"
```

### Step 2.2 — Migrate to Ray Serve

```bash
# Deploy Ray Serve model
python -c "from backend.app.serving.ray_serve_config import deploy_model; deploy_model()"

# Enable in API
export RAY__ENABLED=true
export RAY__SERVE_URL=http://localhost:8000

# Verify Ray Dashboard
open http://localhost:8265
```

### Step 2.3 — Enable Kafka Audit Logs

```bash
# Start Kafka
docker compose -f docker-compose.yml -f docker-compose.2026.yml up -d kafka zookeeper

# Enable in API
export KAFKA__ENABLED=true
export KAFKA__BOOTSTRAP_SERVERS=localhost:9092

# Verify audit events flowing
# (JSONL fallback continues to work alongside Kafka)
```

### Step 2.4 — Enable mTLS (Production Only)

```bash
# Generate dev certificates
python -c "from backend.app.core.mtls import create_dev_certificates; create_dev_certificates()"

# Enable
export MTLS__ENABLED=true
export MTLS__CA_CERT_PATH=certs/ca.pem
export MTLS__CLIENT_CERT_PATH=certs/client.pem
export MTLS__CLIENT_KEY_PATH=certs/client.key
```

### Step 2.5 — Generate SBOM

```bash
# Build image and generate SBOM
chmod +x scripts/generate_sbom.sh
./scripts/generate_sbom.sh retinalai:latest
```

### Phase 2 Rollback

```bash
export RAY__ENABLED=false      # Falls back to local ModelService
export KAFKA__ENABLED=false    # Falls back to JSONL audit
export MTLS__ENABLED=false     # Disables mTLS
```

---

## Phase 3: Governance, Fairness & Edge

### Step 3.1 — Export Edge Models

```bash
# Export to all formats
python scripts/export_all_formats.py \
  --model-path models/model_vignn_rank1.pth \
  --output-dir models/export \
  --formats onnx,torchscript,int8
```

### Step 3.2 — Enable Edge Endpoints

```bash
export EDGE__ONNX_ENABLED=true
export EDGE__ONNX_MODEL_PATH=models/export/model.onnx

# Verify edge inference
curl -X POST http://localhost:8080/api/v1/predict/onnx \
  -F "file=@test_image.jpg"
```

### Step 3.3 — Enable Fairness Dashboard

```bash
export FAIRNESS__ENABLED=true
curl http://localhost:8080/api/v1/governance/fairness
```

### Step 3.4 — Enable Auto Model Cards

```bash
export MODEL_CARD__AUTO_GENERATE=true
curl http://localhost:8080/api/v1/governance/model-card
```

### Phase 3 Rollback

```bash
export EDGE__ONNX_ENABLED=false
export FAIRNESS__ENABLED=false
export MODEL_CARD__AUTO_GENERATE=false
```

---

## Phase 4: Future-Proofing

### Step 4.1 — Enable Graceful Degradation

```bash
export RESILIENCE__ENABLED=true
# System automatically degrades when services become unhealthy
```

### Phase 4 Rollback

```bash
export RESILIENCE__ENABLED=false
export MULTIMODAL__ENABLED=false
export FEDERATED__ENABLED=false
```

---

## Full Stack Deployment (All Phases)

```bash
# Install all dependencies
pip install -e ".[full]"

# Start full infrastructure
docker compose -f docker-compose.yml -f docker-compose.2026.yml up -d

# Or use Makefile
make up-full
```

## Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f k8s/base/

# Deploy backend
kubectl apply -f k8s/base/backend-deployment.yaml

# Run chaos experiments (after LitmusChaos operator is installed)
kubectl apply -f k8s/chaos/litmuschaos-manifests.yaml
```

## Makefile Targets

```bash
make up-phase1    # Phase 1 infrastructure (OTEL + Jaeger + Prometheus + MLflow)
make up-phase2    # Phase 1 + 2 infrastructure (+ Ray Serve + Kafka)
make up-full      # Full 2026 stack (all phases)
make down-full    # Teardown full stack
make test         # Run 198 tests
make sbom         # Generate SBOM (Syft + Grype)
make export-all   # Export model to all formats
```

## Troubleshooting

### Feature not activating

All features use `env_nested_delimiter="__"`. Ensure your `.env` uses double-underscore:
```bash
# Correct:
TELEMETRY__ENABLED=true

# Wrong (single underscore):
TELEMETRY_ENABLED=true
```

### MLflow connection refused

Ensure the MLflow container is running and healthy:
```bash
docker compose -f docker-compose.yml -f docker-compose.mlflow.yml ps
curl http://localhost:5000/health
```

### OTEL traces not appearing in Jaeger

1. Verify OTEL Collector is running: `curl http://localhost:13133/`
2. Check collector config: `configs/otel-collector-config.yaml`
3. Ensure the API has `TELEMETRY__OTLP_ENDPOINT` pointing to the collector

### Active learning not triggering fine-tune

The threshold defaults to 150 corrected samples. Check progress:
```bash
curl http://localhost:8080/api/v1/governance/active-learning-stats
# Look for "progress": "42/150"
```

### Edge endpoint returns 501

The edge format must be loaded. Verify:
```bash
curl http://localhost:8080/api/v1/predict/edge/status
# Check loaded_formats array
```

Export the model first:
```bash
make export-all
export EDGE__ONNX_ENABLED=true
export EDGE__ONNX_MODEL_PATH=models/export/model.onnx
```

## Environment Variable Reference

See [`configs/backend_2026.yaml`](../configs/backend_2026.yaml) for the complete list of all environment variables with defaults and descriptions.

## Architecture Decision Records

| Decision | Rationale |
|----------|-----------|
| Nested Pydantic settings with `__` delimiter | Groups 60+ env vars into logical sections; backward compatible with flat vars |
| No-op pattern for disabled features | Zero import/runtime cost when disabled; callers don't need `if enabled:` checks |
| Event bus dual transport (in-process + Kafka) | Kafka adds durability without changing in-process coordination |
| JSONL audit fallback | Ensures compliance logging continues if Kafka is down |
| Lazy imports for heavy deps | OTEL, MLflow, Ray, Kafka add ~500ms startup each; only pay when enabled |
| Circuit breaker per external service | Prevents cascading failures; separate thresholds for Claude vs Groq vs Ray |
| CRC-32 consistent hashing for canary | Lightweight, deterministic per request_id for sticky sessions |
| LoRA for active learning fine-tuning | <5M trainable params on a 25M model; fast fine-tune, minimal catastrophic forgetting |

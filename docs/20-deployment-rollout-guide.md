# Deployment & Rollout Guide

> RetinalAI Phase 5: Offline-First, Mobile-First, Voice-First Deployment

## Phased Rollout Plan

### Stage 1: Internal Testing (Weeks 1-2)
- Deploy quantized models to staging environment
- Run automated quality gates (faithfulness, WER, bundle size, latency)
- Internal team tests offline RAG + voice-first features
- **Gate**: All acceptance criteria pass in staging

### Stage 2: Pilot Regions (Weeks 3-6)
- Select 2-3 pilot health facilities in Kampala (good connectivity for fallback)
- Deploy Flutter APK to 20-30 community health workers
- Monitor offline usage, sync reliability, voice accuracy
- Collect user feedback on voice-first UX
- **Gate**: >= 85% user satisfaction, offline faithfulness >= 0.82

### Stage 3: Rural Expansion (Weeks 7-12)
- Expand to 10 rural health facilities
- Test on 2G/3G networks and fully offline scenarios
- Monitor bundle download completion rate
- Verify delta sync works with intermittent connectivity
- **Gate**: >= 90% successful offline screenings, WER <= 18%

### Stage 4: Nationwide Deployment (Weeks 13+)
- Full rollout to all partner facilities
- Enable voice-first as default mobile interface
- Open Google Play Store listing (if applicable)
- **Gate**: All acceptance criteria met, < 2% error rate

## Docker & Kubernetes Updates

### Quantized Model Serving

```bash
# Standard deployment (existing)
docker compose up -d

# With quantized models + torch.compile
QUANTIZATION__ENABLED=true \
QUANTIZATION__ACTIVE_FORMAT=gguf_q4_k_m \
QUANTIZATION__TORCH_COMPILE_ENABLED=true \
docker compose up -d

# With offline RAG
OFFLINE_RAG__ENABLED=true \
docker compose up -d

# Full Phase 5 stack
make up-full-v2
```

### Production vLLM GPU Profile

The GPU 7 vLLM service now uses the AWQ-quantized `Qwen/Qwen3-8B-AWQ` checkpoint with an 8K context window and a right-sized KV cache:

```bash
docker run -d \
  --name ura-vllm \
  --gpus '"device=7"' \
  -p 8011:8001 \
  -v /home/developer/.cache/huggingface:/root/.cache/huggingface \
  -v /home/developer/models/huggingface:/root/models/huggingface \
  vllm/vllm-openai:v0.8.5 \
  --model Qwen/Qwen3-8B-AWQ \
  --download-dir /root/models/huggingface \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --quantization awq \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Validated result on 2026-05-07: GPU 7 dropped from 41.1 GB used to 15.5 GB used, model weights dropped from 16.4 GB BF16 to 5.7 GB AWQ INT4, and vLLM reported a 31,424-token KV cache with 3.84x concurrency for full 8,192-token requests. This frees about 25.6 GB of VRAM for CosyVoice2, Qwen2-VL, and other companion services.

See [vLLM GPU 7 AWQ Optimization Runbook](21-vllm-gpu7-awq-optimization.md) for the tuning history, verification commands, and rollback profile.

### Kubernetes Deployment Update

Add to `k8s/base/backend-deployment.yaml`:

```yaml
# Phase 5 environment variables (add to containers[].env)
- name: QUANTIZATION__ENABLED
  valueFrom:
    configMapKeyRef:
      name: retinalai-phase5
      key: quantization_enabled
- name: QUANTIZATION__ACTIVE_FORMAT
  valueFrom:
    configMapKeyRef:
      name: retinalai-phase5
      key: quantization_format
- name: OFFLINE_RAG__ENABLED
  valueFrom:
    configMapKeyRef:
      name: retinalai-phase5
      key: offline_rag_enabled
- name: VOICE_FIRST__ENABLED
  valueFrom:
    configMapKeyRef:
      name: retinalai-phase5
      key: voice_first_enabled

# Phase 5 volume mounts (add to containers[].volumeMounts)
- name: offline-rag-data
  mountPath: /app/data/offline_rag
- name: quantized-models
  mountPath: /app/outputs/quantized

# Phase 5 volumes (add to volumes)
- name: offline-rag-data
  persistentVolumeClaim:
    claimName: retinalai-offline-rag-pvc
- name: quantized-models
  persistentVolumeClaim:
    claimName: retinalai-quantized-models-pvc
```

ConfigMap for Phase 5:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: retinalai-phase5
data:
  quantization_enabled: "false"
  quantization_format: ""
  offline_rag_enabled: "false"
  voice_first_enabled: "false"
  mobile_bundle_enabled: "false"
```

### Resource Requirements

```yaml
# Quantized model serving (reduced from baseline)
resources:
  requests:
    memory: "1Gi"      # Was 1.8Gi with bfloat16
    cpu: "500m"
  limits:
    memory: "2Gi"      # Was 3Gi
    nvidia.com/gpu: "1"
```

## Flutter Build Instructions

### Prerequisites

```bash
# Install Flutter SDK (3.22+)
flutter --version

# Install Android SDK (API 21+)
sdkmanager --install "platforms;android-34" "build-tools;34.0.0"

# Generate Flutter scaffold
make flutter-scaffold

# Build mobile bundle with voice models
make mobile-bundle
```

### Build Offline-Capable APK

```bash
cd outputs/mobile_bundle/MobileApp

# Install dependencies
flutter pub get

# Copy model assets
cp -r ../flutter_assets/models/ assets/models/

# Build release APK
flutter build apk --release --target-platform android-arm64

# Build AAB for Play Store
flutter build appbundle --release
```

### Bundle Size Verification

```bash
# Check APK size
ls -lh build/app/outputs/flutter-apk/app-release.apk

# Detailed size analysis
flutter build apk --analyze-size --target-platform android-arm64

# Validate against 800 MB limit
make mobile-bundle-validate
```

## Rollback Strategy

### Instant Switch to Online-Only Mode

All Phase 5 features are behind feature flags. Rollback is instant:

```bash
# Disable all Phase 5 features
OFFLINE_RAG__ENABLED=false \
QUANTIZATION__ENABLED=false \
VOICE_FIRST__ENABLED=false \
MOBILE_BUNDLE__ENABLED=false \
docker compose up -d

# Or in Kubernetes
kubectl edit configmap retinalai-phase5
# Set all values to "false"
kubectl rollout restart deployment/retinalai-backend
```

### Per-Feature Rollback

| Feature | Rollback Command | Impact |
|---------|-----------------|--------|
| Quantization | `QUANTIZATION__ENABLED=false` | Falls back to bfloat16 model, higher memory usage |
| Offline RAG | `OFFLINE_RAG__ENABLED=false` | Offline endpoints return 503, mobile falls back to online |
| Voice-First | `VOICE_FIRST__ENABLED=false` | Voice UI hidden, text-only interface |
| Mobile Bundle | `MOBILE_BUNDLE__ENABLED=false` | Bundle download endpoints disabled |

### Data Safety

- Offline data stored on-device is never deleted during rollback
- Pending sync queue is preserved and processed when feature is re-enabled
- Audit logs for offline decisions are retained in local storage
- No user data is lost during any rollback scenario

## Production Readiness Checklist

### Quantization & Performance
- [ ] GGUF Q4_K_M faithfulness >= 0.89 (drop <= 4% from bfloat16)
- [ ] Server p95 latency <= 1.8s for full RAG pipeline
- [ ] Server memory reduction >= 38% (measured via Prometheus)
- [ ] CI pipeline produces GGUF + AWQ + ONNX artifacts automatically
- [ ] Quality gates block merge on threshold violations
- [ ] `GET /api/v1/models/quantized` returns available variants

### Offline RAG
- [ ] Offline bundle <= 150 MB compressed
- [ ] Offline faithfulness >= 0.82 on 50 test queries
- [ ] Delta sync < 12 seconds for typical daily changes
- [ ] SHA-256 integrity verification passes for all bundles
- [ ] User can toggle "Offline Mode" and get consistent results
- [ ] `GET /api/v1/offline/status` returns healthy

### Mobile Bundle
- [ ] Total Flutter + model bundle <= 800 MB
- [ ] On-device vector search < 180ms p95 on 4GB RAM Android
- [ ] App fully functional offline (common questions answered)
- [ ] Bundle download completes successfully on 3G connection
- [ ] CI enforces 800 MB bundle size limit

### Voice-First Mobile
- [ ] Voice chat p95 latency < 1.2s (online), < 2.0s (offline)
- [ ] Barge-in success rate >= 92%
- [ ] Offline speech WER <= 18% on Ugandan English test set
- [ ] Voice + vision mode < 3s end-to-end latency
- [ ] Touch targets >= 48px for all interactive elements

### Governance & Quality
- [ ] All features behind feature flags (FLAG_QUANTIZATION, FLAG_OFFLINE_RAG, etc.)
- [ ] Full audit trail for offline vs online decisions
- [ ] Model Card updated with quantization, offline, and mobile sections
- [ ] Prometheus metrics exported (`retinalai_offline_*`, `retinalai_voice_*`, etc.)
- [ ] Grafana dashboard "Offline & Mobile Experience" configured
- [ ] 100% new code has unit + integration tests (coverage >= 80%)

### Security & Privacy
- [ ] No raw patient data leaves device without explicit consent
- [ ] Offline inference results are anonymized before sync
- [ ] Bundle integrity verified before use (SHA-256)
- [ ] mTLS between mobile app and API (when online)
- [ ] Rate limiting applied to sync and bundle download endpoints

### Monitoring
- [ ] `GET /api/v1/admin/offline_stats` returns valid data
- [ ] `GET /api/v1/admin/voice_stats` returns valid data
- [ ] `GET /api/v1/admin/stats` aggregates all metrics
- [ ] `GET /api/v1/admin/metrics/prometheus` returns Prometheus format
- [ ] Alerts configured for: offline faithfulness drop, sync failures, bundle size exceeded

## New Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/models/quantized` | List available quantized model variants |
| GET | `/api/v1/models/optimization/status` | Server optimization status |
| GET | `/api/v1/offline/status` | Offline RAG pipeline status |
| POST | `/api/v1/offline/sync` | Trigger delta sync |
| GET | `/api/v1/offline/bundle` | Download current bundle |
| GET | `/api/v1/offline/bundle/info` | Bundle metadata |
| GET | `/api/v1/offline/bundle/versions` | List bundle versions |
| POST | `/api/v1/offline/bundle/verify` | Verify bundle integrity |
| POST | `/api/v1/offline/search` | Offline RAG search |
| GET | `/api/v1/admin/offline_stats` | Offline usage statistics |
| GET | `/api/v1/admin/voice_stats` | Voice-first statistics |
| GET | `/api/v1/admin/quantization_stats` | Quantization performance |
| GET | `/api/v1/admin/stats` | Full admin dashboard |
| GET | `/api/v1/admin/metrics/prometheus` | Prometheus metrics export |
| GET | `/api/v1/admin/grafana/dashboard` | Grafana dashboard config |

## Make Targets Summary

```bash
# Quantization
make quantize              # GGUF + ONNX (default)
make quantize-all          # All formats (GGUF + AWQ + GPTQ + ONNX + TRT)
make quality-gate          # Run quality gates on quantized models

# Mobile
make mobile-bundle         # Build mobile bundle with voice models
make mobile-bundle-validate # Validate bundle size against 800 MB limit
make flutter-scaffold      # Generate Flutter project structure

# Docker
make up-offline            # API with offline RAG enabled
make up-quantized          # API with quantized models + torch.compile
make up-full-v2            # Full Phase 5 stack
```

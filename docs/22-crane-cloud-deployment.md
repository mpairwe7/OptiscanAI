# 22. Crane Cloud Deployment Guide

Production deployment of OptiscanAI on [Crane Cloud](https://cranecloud.io) — Uganda's Kubernetes-as-a-Service platform.

> **Last verified:** 2026-05-15
> **Canonical SaaS URL:** https://www.optiscan.makstartup.com
> **Live CPU URL:** https://optiscan-ai-4fe6e1aa.renu-01.cranecloud.io
> **Live GPU (CPU fallback) URL:** https://optiscan-gpu-renu-9458f363.renu-01.cranecloud.io
> **Live CPU Fallback URL:** https://optiscan-cpu-fallback-e4d13933.renu-01.cranecloud.io
> **Docker Hub:** https://hub.docker.com/r/landwind/optiscan-ai

---

## SaaS billing layer — Postgres requirement

When `BILLING__ENABLED=true` the backend needs a Postgres reachable at
`$DATABASE__URL`. The current default ships **Option 2 — sidecar
Postgres** for both compose and Kubernetes. Bring up the database before
the API rolls out:

```bash
# Replace POSTGRES_PASSWORD in postgres-secret.yaml first!
kubectl apply -f k8s/base/postgres-secret.yaml
kubectl apply -f k8s/base/postgres-service.yaml
kubectl apply -f k8s/base/postgres-statefulset.yaml
kubectl rollout status statefulset/optiscan-postgres -n retinalai
kubectl apply -f k8s/base/backend-deployment.yaml
```

The StatefulSet provisions a 20 Gi `PersistentVolumeClaim` (default
StorageClass — override to `cranecloud-ssd` via a kustomize patch if your
project has it).

If you'd rather not run a sidecar (single-pod pilot, tight memory budget):
set `EMBEDDED_POSTGRES__ENABLED=true` on the api Deployment and the in-image
Postgres takes over. **Mount a PVC at `/var/lib/postgresql/data` in that
case** — without it, every pod restart resets the database.

See [docs/23-billing-platform.md](23-billing-platform.md) §§ 14–18 for the
full architecture and runbook.

---

## Docker Images

Two images are published to Docker Hub under `landwind/optiscan-ai`:

| Tag | Base Image | Size | PyTorch | Model Weights | Use Case |
|-----|-----------|------|---------|---------------|----------|
| `cpu` | `python:3.11-slim-bookworm` | **2.4 GB** | CPU-only (`whl/cpu`) | Baked in (137 MB) | Crane Cloud (RENU), any CPU host |
| `latest` | `nvidia/cuda:12.4.1-cudnn-runtime` | **10.1 GB** | CUDA 12.4 (`cu124`) | Baked in (137 MB) | GPU servers, Crane Cloud (AHUMAIN) |

Both images are **full-stack**: nginx (port 8080) + FastAPI backend (8081) + Next.js frontend (3000), orchestrated by supervisord.

### Docker Hub Credentials

| Field | Value |
|-------|-------|
| Registry | `hub.docker.com` |
| Username | `landwind` |
| Access Token | Stored as GitHub Secret `DOCKERHUB_TOKEN` |

Pull commands:
```bash
docker pull landwind/optiscan-ai:cpu      # Crane Cloud / CPU
docker pull landwind/optiscan-ai:latest   # GPU servers
```

---

## Architecture (Single Container)

```
Port 8080 (nginx reverse proxy)
  ├── /              → Next.js frontend UI (port 3000)
  ├── /api/          → FastAPI backend (port 8081)
  ├── /v1/           → Phase 5 APIs (offline, voice, quantized)
  ├── /health        → Backend health check
  ├── /docs          → Swagger API documentation
  ├── /openapi.json  → OpenAPI spec
  └── /_next/static/ → Static assets (nginx-cached, 365d)
```

Internal process management (supervisord):
```
supervisord (PID 1)
  ├── [program:backend]   uvicorn backend.app.main:app --host 127.0.0.1 --port 8081
  ├── [program:frontend]  node server.js  (Next.js standalone, port 3000)
  └── [program:nginx]     nginx -g "daemon off;"  (port 8080)
```

Each process has `autorestart=true`, `stopsignal=TERM` (QUIT for nginx), and `stopwaitsecs` for graceful shutdown.

---

## Crane Cloud Platform Reference

### Available Clusters (as of 2026-05-12)

| Cluster | ID | Subdomain | Status | GPU Support | Notes |
|---------|-----|-----------|--------|-------------|-------|
| **RENU** | `9e81a70e-8460-4e5d-b0a8-17abcac30f68` | `renu-01.cranecloud.io` | Active | No | Recommended — stable, fast pod scheduling |
| **AHUMAIN ML** | `df2eeac2-b36d-4bbd-a734-eb03754cd175` | `ahumain.cranecloud.io` | Active | `supports_ml: true` | Pod scheduling issues observed (pods stuck in "unknown") |
| **Makerere-1** | `f3068db2-a981-4308-8c57-64112a792365` | `cranecloud.io` | Disabled | No | Legacy cluster, not available |

### Crane Cloud API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/login` | POST | Login (returns `access_token`) |
| `/projects` | GET | List projects |
| `/projects` | POST | Create project (requires `cluster_id`, `owner_id`) |
| `/projects/{id}/apps` | GET | List apps in project |
| `/projects/{id}/apps` | POST | Deploy app |
| `/apps/{id}` | GET | App details (status, pods, env vars) |
| `/apps/{id}` | PATCH | Update app (env vars, image, replicas) |
| `/apps/{id}` | DELETE | Delete app |
| `/clusters` | GET | List clusters |

> **Note:** Email is case-sensitive for login. Use lowercase (e.g., `mpairwelauben75@gmail.com`).

### Crane Cloud CLI Keyring Issue

On headless servers (no D-Bus secret storage), the `cranecloud` CLI crashes with `keyring.errors.InitError`. Workaround:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
cranecloud auth login
```

Or bypass the CLI and use the API directly (see deployment scripts below).

---

## CPU Deployment (RENU Cluster)

### Why CPU for Crane Cloud?

1. **RENU has no NVIDIA GPUs** — `torch.cuda.is_available()` returns `False`, auto-falls back to CPU
2. **AHUMAIN ML has scheduling issues** — pods stuck in "unknown" status for hours (cluster-level, not image/config)
3. **Image size matters** — 2.4 GB pulls in ~60s vs 10.1 GB taking 4+ minutes
4. **CPU inference is fast enough** — ~87ms inference + ~46ms gate after warmup (verified 2026-05-12)
5. **GPU image also works on CPU** — `DEVICE=auto` detects no CUDA and falls back gracefully
6. **Crane Cloud golden rule** — model weights must be baked in (no volume mounts)

### Model Weights (Baked In)

Crane Cloud has **no volume mount support**. Both images bake weights at build time:

```dockerfile
COPY models/model_vignn_rank1.pth ./models/model_vignn_rank1.pth   # 131 MB (ViGNN classifier)
COPY weights/fundus_gate.pth ./weights/fundus_gate.pth             # 6 MB   (MobileNetV3 gate)
```

> **Critical:** Do NOT rely on volume mounts (`-v`) or runtime model downloads.
> Crane Cloud's health checks terminate the container before downloads complete.

### Deployment Steps

#### Option A: Crane Cloud Web UI

1. Log in at [cranecloud.io](https://cranecloud.io)
2. Create project → select **RENU** cluster
3. Create app:
   - **Image URI:** `landwind/optiscan-ai:cpu`
   - **Port:** `8080`
   - **Replicas:** `1`
   - **Entry Command:** *(leave empty)*
4. Add environment variables (no spaces in keys or values):

| Key | Value | Purpose |
|-----|-------|---------|
| `MODEL_PATH` | `models/model_vignn_rank1.pth` | ViGNN model checkpoint |
| `CUDA_VISIBLE_DEVICES` | `-1` | Disable GPU |
| `DEVICE` | `cpu` | Force CPU inference |
| `FUNDUS_GATE__ENABLED` | `true` | Enable image quality gate |

#### Option B: Crane Cloud API

```python
import requests, json

API = "https://api.cranecloud.io"

# Step 1: Login
resp = requests.post(f"{API}/users/login", json={
    "email": "mpairwelauben75@gmail.com",   # must be lowercase
    "password": "your-password"
})
token = resp.json()["data"]["access_token"]
user_id = resp.json()["data"]["id"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Step 2: Create project on RENU cluster
resp = requests.post(f"{API}/projects", headers=headers, json={
    "name": "OptiscanAI",
    "cluster_id": "9e81a70e-8460-4e5d-b0a8-17abcac30f68",  # RENU
    "owner_id": user_id
})
project_id = resp.json()["data"]["project"]["id"]

# Step 3: Deploy CPU app
resp = requests.post(f"{API}/projects/{project_id}/apps", headers=headers, json={
    "name": "optiscan-ai",
    "image": "landwind/optiscan-ai:cpu",
    "port": 8080,
    "replicas": 1,
    "env_vars": {
        "MODEL_PATH": "models/model_vignn_rank1.pth",
        "CUDA_VISIBLE_DEVICES": "-1",
        "DEVICE": "cpu",
        "FUNDUS_GATE__ENABLED": "true"
    }
})
app = resp.json()["data"]["app"]
print(f"URL: {app['url']}")

# Step 4: Monitor status
import time
for i in range(20):
    r = requests.get(f"{API}/apps/{app['id']}", headers=headers)
    status = r.json()["data"]["apps"]["app_running_status"]
    pods = r.json()["data"]["apps"].get("pod_statuses", [])
    print(f"[{i}] {status} | pods: {pods}")
    if status == "running":
        print(f"LIVE: {app['url']}")
        break
    time.sleep(15)
```

### Verified CPU Deployment (2026-05-12)

| Field | Value |
|-------|-------|
| Cluster | RENU |
| App name | optiscan-ai |
| App ID | `f58201b7-5ba1-48b1-a635-02ee29965352` |
| URL | https://optiscan-ai-4fe6e1aa.renu-01.cranecloud.io |
| Image | `landwind/optiscan-ai:cpu` |
| Status | Running |
| Pod startup | `waiting` → `running` → healthy in ~90 seconds |

Endpoints verified:

| Endpoint | Status | Response |
|----------|--------|----------|
| `/health` | 200 | `{"status": "healthy", "model_loaded": true, "device": "cpu", "diseases_count": 45}` |
| `/` | 200 | Next.js frontend UI |
| `/docs` | 200 | Swagger API documentation |
| `/api/v1/gate/status` | 200 | Gate v2 enabled, learned model loaded |
| `/api/v1/predict` | 422 | Correctly rejects non-fundus images |
| `/api/v1/system/info` | 200 | Platform info, PyTorch version |
| `/api/v1/diseases` | 200 | Disease list |

---

## GPU / CUDA Deployment

### When to Use GPU Image

- Dedicated NVIDIA GPU server (e.g., RTX A6000, T4, A100)
- Cloud VMs with NVIDIA drivers (AWS p3/g4, GCP A2, Azure NC)
- Crane Cloud AHUMAIN ML cluster (when scheduling stabilizes)

### GPU Image Specifications

| Component | Version |
|-----------|---------|
| Base | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| PyTorch | `2.6.0+cu124` |
| CUDA | 12.4 |
| cuDNN | 9.1.0.70 |
| Triton | 3.2.0 |
| Python | 3.11 (deadsnakes PPA) |

### Running on Local GPU Server

```bash
docker run -d \
  --name optiscan-gpu \
  --gpus '"device=0"' \
  -p 8080:8080 \
  landwind/optiscan-ai:latest

# Verify CUDA inference
curl http://localhost:8080/health
# {"status": "healthy", "model_loaded": true, "device": "cuda", "diseases_count": 45}

curl http://localhost:8080/api/v1/system/info | jq .infrastructure
# {"cuda_available": true, "cuda_version": "12.4", "gpu": "NVIDIA RTX A6000", "gpu_memory": "47.5 GB", "device": "cuda"}
```

### Verified Local GPU Inference (2026-05-12)

Tested on local server with 8x NVIDIA RTX A6000 (49 GB each):

| Metric | GPU (CUDA) | CPU (Crane Cloud RENU) |
|--------|-----------|------------------------|
| ViGNN inference | **148ms** | 87ms |
| Fundus gate v2 | **125ms** | 46ms |
| Total pipeline | **287ms** | 133ms |
| `torch.cuda.is_available()` | `True` | `False` |
| Device reported | `cuda` | `cpu` |

> GPU inference is ~2x faster on first cold-start prediction (~700ms GPU vs ~800ms CPU)
> but both converge after warmup. GPU advantage grows with batch size and model complexity.

### Running with docker-compose

```bash
docker compose up -d          # GPU backend (default)
docker compose --profile cpu up  # CPU backend
docker compose --profile hf up   # HF Spaces (port 7860)
```

### Crane Cloud GPU Deployment (AHUMAIN ML Cluster)

The AHUMAIN cluster advertises `supports_ml: true`. When operational:

```python
# Create project on AHUMAIN
requests.post(f"{API}/projects", headers=headers, json={
    "name": "OptiscanAI-GPU",
    "cluster_id": "df2eeac2-b36d-4bbd-a734-eb03754cd175",  # AHUMAIN ML
    "owner_id": user_id
})

# Deploy GPU app with auto device detection
requests.post(f"{API}/projects/{project_id}/apps", headers=headers, json={
    "name": "optiscan-gpu",
    "image": "landwind/optiscan-ai:latest",
    "port": 8080,
    "replicas": 1,
    "env_vars": {
        "MODEL_PATH": "models/model_vignn_rank1.pth",
        "CUDA_VISIBLE_DEVICES": "0",
        "DEVICE": "auto",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "FUNDUS_GATE__ENABLED": "true"
    }
})
```

#### GPU Image on RENU (Auto CPU Fallback)

RENU has no NVIDIA GPUs. The GPU image (`latest`) deploys and runs correctly with `DEVICE=auto` — PyTorch auto-detects no CUDA and falls back to CPU inference. No manual override needed.

| Key | Recommended Value | Behavior |
|-----|-------------------|----------|
| `DEVICE` | `auto` | Detects GPU if available, falls back to CPU |
| `CUDA_VISIBLE_DEVICES` | `0` (GPU) or `-1` (force CPU) | Controls GPU visibility |

> The GPU image on a CPU-only cluster wastes ~7 GB of CUDA libraries. Use the `:cpu` tag for production on CPU-only clusters.

### Verified Deployments (2026-05-12)

#### RENU Cluster (all healthy)

| App | Image | Device | App ID | URL | Status |
|-----|-------|--------|--------|-----|--------|
| optiscan-ai | `:cpu` | cpu | `f58201b7-5ba1-48b1-a635-02ee29965352` | `optiscan-ai-4fe6e1aa.renu-01.cranecloud.io` | **Running** |
| optiscan-cpu-fallback | `:cpu` | cpu | `cef474cf-0805-4b0c-9cbc-7b27fe74bd4f` | `optiscan-cpu-fallback-e4d13933.renu-01.cranecloud.io` | **Running** |
| optiscan-gpu-renu | `:latest` | cpu (auto) | `8efaf32a-c9be-4bf8-9fe3-f79d01940e88` | `optiscan-gpu-renu-9458f363.renu-01.cranecloud.io` | **Running** |

#### AHUMAIN ML Cluster (broken)

| App | Image | App ID | Status | Notes |
|-----|-------|--------|--------|-------|
| optiscan-gpu | `:cpu` | `90c50d78-ef62-4b4b-896c-4ad34e620e5c` | **Failed** | Pod unknown — even CPU image fails |

### AHUMAIN Cluster Issues (2026-05-12)

Multiple deployment attempts on AHUMAIN failed with:
- Pod status: `unknown` (not `pending`, `waiting`, or `running`)
- Failure persists even with the lightweight 2.4 GB CPU image
- Confirmed cluster-level issue, not image or config

**Root cause:** Cluster-level scheduling/node issue. Recommendation: use RENU until AHUMAIN stabilizes. Contact Crane Cloud support.

---

## CUDA Configuration Reference

### Environment Variables

| Variable | CPU Value | GPU Value | Description |
|----------|-----------|-----------|-------------|
| `CUDA_VISIBLE_DEVICES` | `-1` | `0` | GPU device index (-1 disables CUDA) |
| `DEVICE` | `cpu` | `cuda` | PyTorch device selection |
| `MODEL_PATH` | `models/model_vignn_rank1.pth` | Same | Path to ViGNN checkpoint |
| `FUNDUS_GATE__ENABLED` | `true` | `true` | Enable MobileNetV3 image quality gate |
| `FUNDUS_GATE__MODEL_PATH` | `weights/fundus_gate.pth` | Same | Gate model weights |

### PyTorch Device Fallback Logic

The backend (`backend/app/core/model_service.py`) handles device selection:

```python
# model_service.py — ViGNN classifier
device_str = settings.device  # "auto" | "cpu" | "cuda"
if device_str == "auto":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
self.device = torch.device(device_str)
```

The fundus gate (`src/data/fundus_gate_learned.py`) auto-detects device independently:

```python
# fundus_gate_learned.py — MobileNetV3 gate
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self.to(self._device)  # moves model to detected device
# tensors also moved: tensor.to(self._device) in check()
```

Both models auto-detect and use the same device. On CPU-only hosts, both run on CPU. On GPU hosts, both run on CUDA. No manual override needed when `DEVICE=auto`.

> **Fixed 2026-05-12**: The fundus gate previously always ran on CPU regardless of GPU availability, causing a 42x performance regression (5,355ms vs 125ms) due to CPU/GPU context switching. Now auto-detects device.

---

## Building Images

```bash
# CPU image (Crane Cloud)
make docker-build-cpu
# or: docker build -t landwind/optiscan-ai:cpu -f Dockerfile.cpu .

# GPU image (NVIDIA servers)
make docker-build
# or: docker build -t landwind/optiscan-ai:latest -f Dockerfile .

# Push individual
make docker-push          # GPU only
make docker-push-cpu      # CPU only
make docker-push-all      # Both
```

### Build Comparison

| Metric | CPU Image | GPU Image |
|--------|-----------|-----------|
| Base image | 150 MB | 3.6 GB |
| PyTorch | ~300 MB (CPU wheels) | ~5.2 GB (CUDA wheels) |
| App + frontend + nginx | ~350 MB | ~350 MB |
| Model weights | 137 MB | 137 MB |
| **Total** | **~2.4 GB** | **~10.1 GB** |
| Build time (cached) | ~2 min | ~5 min |
| Build time (clean) | ~5 min | ~12 min |
| Docker Hub push | ~1 min | ~5 min |
| Crane Cloud pull | ~60s | ~4 min |

---

## CI/CD Pipeline

GitHub Actions workflow `.github/workflows/docker-publish.yml` — full MLOps pipeline:

```
push to main → test (lint + pytest) → build & push (GPU + CPU) → deploy to Crane Cloud
```

### Pipeline Stages

| Stage | Job | What it does |
|-------|-----|-------------|
| 1. Test | `test` | Ruff lint, Black format check, pytest (CPU) |
| 2. Build | `build-and-push` | Matrix: GPU (`Dockerfile` → `:latest`) + CPU (`Dockerfile.cpu` → `:cpu`) |
| 3. Deploy | `deploy-crane-cloud` | Login to Crane Cloud API, PATCH both apps to trigger image pull |
| 4. Verify | Health check | Poll CPU app `/health` for up to 5 minutes |

### Triggers

| Trigger | Behavior |
|---------|----------|
| Push to `main` (Dockerfile, src/, backend/, frontend/, configs/) | Full pipeline: test → build → deploy |
| Git tag `v*` | Full pipeline with version tag |
| Manual dispatch | Full pipeline or deploy-only (skip build) |

### Deploy-Only Mode

To redeploy without rebuilding (e.g., env var change):

1. Go to **Actions** → **CI/CD — Build, Push & Deploy** → **Run workflow**
2. Check **"Skip build, only redeploy Crane Cloud"**
3. Click **Run workflow**

### GitHub Secrets Required

Set at `github.com/mpairwe7/MLOPS_V1/settings/secrets/actions`:

| Secret | Description | Example |
|--------|-------------|---------|
| `DOCKERHUB_USERNAME` | Docker Hub username | `landwind` |
| `DOCKERHUB_TOKEN` | Docker Hub access token | `dckr_pat_...` |
| `CRANE_CLOUD_EMAIL` | Crane Cloud login email (lowercase) | `mpairwelauben75@gmail.com` |
| `CRANE_CLOUD_PASSWORD` | Crane Cloud login password | |
| `CRANE_CLOUD_CPU_APP_ID` | CPU app ID on RENU | `f58201b7-5ba1-48b1-a635-02ee29965352` |
| `CRANE_CLOUD_GPU_APP_ID` | GPU (latest) app ID on RENU | `8efaf32a-c9be-4bf8-9fe3-f79d01940e88` |
| `CRANE_CLOUD_CPU_URL` | CPU app URL for health check | `https://optiscan-ai-4fe6e1aa.renu-01.cranecloud.io` |

---

## Docker Best Practices Applied (26/26)

Both Dockerfiles pass all industry standards:

| # | Practice | Implementation |
|---|----------|----------------|
| 1 | Multi-stage build | 3 stages: Python builder, frontend builder, slim runtime |
| 2 | `.dockerignore` | Comprehensive — build context ~20 KB |
| 3 | `--no-install-recommends` | On all `apt-get install` |
| 4 | Clean apt cache | `rm -rf /var/lib/apt/lists/*` |
| 5 | `PIP_NO_CACHE_DIR=1` | ENV + `--no-cache-dir` on all pip install |
| 6 | Pinned versions | `torch==2.6.0+cu124`, exact versions |
| 7 | `HEALTHCHECK` | 30s interval, 60s start-period, 3 retries |
| 8 | `EXPOSE` | 8080 |
| 9 | OCI labels | `org.opencontainers.image.{title,version,source,vendor,licenses}` |
| 10 | Non-root user | `optiscan` user with minimal permissions |
| 11 | Minimal base image | CUDA runtime (not devel) / python-slim |
| 12 | Layer cache ordering | Dependencies before application code |
| 13 | Single-layer user+dirs | No extra chmod layer |
| 14 | Consolidated ENV | Merged into single directive |
| 15 | Specific COPY paths | No `COPY . .` |
| 16 | `STOPSIGNAL SIGTERM` | Graceful shutdown |
| 17 | `SHELL [bash, -euo, pipefail]` | Fail-fast error handling |
| 18 | Frontend layer caching | `package.json` copied before source |
| 19 | Merged RUN commands | Chained apt + cleanup |
| 20 | Minimized layer bloat | Stripped `.dist-info`, `__pycache__`, test dirs |
| 21 | Security scanning | Trivy in CI (`security-scan.yml`) |
| 22 | OCI image labels | Full metadata set |
| 23 | Secure Node.js install | GPG key verification (no `curl \| bash`) |
| 24 | No duplicate deps | PyTorch CUDA installed first, skip re-download |
| 25 | Stripped unused packages | Removed streamlit, pyarrow, altair, pydeck, opencv-python |
| 26 | Graceful shutdown | `stopsignal` + `stopwaitsecs` per supervisord process |

---

## Troubleshooting

### Pod status "unknown" on Crane Cloud

**Symptom:** Pod shows `status: unknown`, `failureReason: unknown` for extended periods.

**Cause:** Cluster-level scheduling issue (observed on AHUMAIN 2026-05-12).

**Fix:**
1. Delete the app: `DELETE /apps/{app_id}`
2. Recreate on RENU cluster instead of AHUMAIN
3. If persistent, contact Crane Cloud support

### `model_loaded: false` in health check

**Symptom:** `/health` returns `{"model_loaded": false, "device": "cuda"}`.

**Cause:** GPU image deployed on CPU-only cluster. CUDA requested but no NVIDIA driver.

**Fix:** Set environment variables:
```
CUDA_VISIBLE_DEVICES=-1
DEVICE=cpu
```

Or use the `:cpu` image tag instead.

### Container exits immediately / CrashLoopBackOff

**Symptom:** Pod restarts repeatedly.

**Cause:** Model weights missing (old image without baked weights).

**Fix:** Verify weights exist in image:
```bash
docker run --rm landwind/optiscan-ai:cpu ls -lh /app/models/ /app/weights/
```
Expected output:
```
/app/models/model_vignn_rank1.pth   131M
/app/weights/fundus_gate.pth        6.0M
```

### Health check timeout / container killed before ready

**Symptom:** Container starts loading but gets killed by Crane Cloud before `/health` returns 200.

**Cause:** Model loading takes 30-40s on CPU. Crane Cloud's default readiness timeout may be shorter.

**Fix:** The Dockerfile has `start-period=60s`. If still failing, reduce model load time by using a smaller checkpoint or pre-warming in a startup script.

### Environment variable keys have spaces

**Symptom:** App ignores env vars even though they appear set.

**Cause:** Leading/trailing spaces in key names (e.g., ` DEVICE ` instead of `DEVICE`). Crane Cloud passes them as-is.

**Fix:** Delete app and recreate, or verify via API:
```python
resp = requests.get(f"{API}/apps/{app_id}", headers=headers)
env = resp.json()["data"]["apps"]["env_vars"]
for k, v in env.items():
    if k != k.strip() or v != v.strip():
        print(f"BAD: key='{k}' value='{v}'")
```

### Image pull too slow / timeout

**Symptom:** Pod stuck in `ContainerCreating` for over 5 minutes.

**Cause:** GPU image is 9.8 GB. Crane Cloud nodes may have bandwidth limits.

**Fix:** Use `:cpu` tag (2.4 GB) — pulls in ~60 seconds on RENU.

---

## Deployment History

| Date | Cluster | Image | Status | URL | Notes |
|------|---------|-------|--------|-----|-------|
| 2026-05-12 | AHUMAIN | `latest` (9.8 GB, no weights) | Failed | `optiscanai-988735ef.ahumain.cranecloud.io` | Pod unknown, env var spaces |
| 2026-05-12 | AHUMAIN | `cpu` (2.1 GB, no weights) | Failed | `optiscanai-5124a665.ahumain.cranecloud.io` | Pod unknown, cluster issue |
| 2026-05-12 | AHUMAIN | `cpu` (2.4 GB, baked weights) | Failed | `optiscanai-17ba7046.ahumain.cranecloud.io` | Pod unknown, cluster issue |
| 2026-05-12 | AHUMAIN | `latest` + GPU env vars | Failed | `optiscan-gpu-3764a139.ahumain.cranecloud.io` | Pod unknown, cluster-level failure |
| 2026-05-12 | AHUMAIN | `cpu` (retest) | Failed | `optiscan-gpu-2d5807bb.ahumain.cranecloud.io` | Confirmed cluster broken, not image |
| 2026-05-12 | RENU | `cpu` (2.4 GB, baked weights) | **Running** | `optiscan-ai-4fe6e1aa.renu-01.cranecloud.io` | All endpoints verified |
| 2026-05-12 | RENU | `cpu` (fallback) | **Running** | `optiscan-cpu-fallback-e4d13933.renu-01.cranecloud.io` | CPU redundancy |
| 2026-05-12 | RENU | `latest` (10.1 GB, DEVICE=auto) | **Running** | `optiscan-gpu-renu-9458f363.renu-01.cranecloud.io` | GPU image, auto-falls back to CPU |

**Lessons learned:**
1. Always bake model weights — Crane Cloud has no volume mounts
2. Always use lowercase email for API login
3. Never put spaces in env var keys/values on Crane Cloud
4. Prefer RENU cluster over AHUMAIN for reliability
5. Use `:cpu` tag for Crane Cloud — 4x smaller, faster pulls, no CUDA dependency
6. PATCH `/apps/{id}` merges env vars — delete and recreate to replace them entirely
7. Use `DEVICE=auto` instead of `DEVICE=cuda` — graceful CPU fallback on clusters without GPUs
8. Fundus gate must auto-detect device (fixed 2026-05-12) — CPU-only gate on GPU host caused 42x regression

---

## Fundus Gate GPU Auto-Detection Fix (2026-05-12)

### Problem

The `LearnedFundusGate` (MobileNetV3-Small) in `src/data/fundus_gate_learned.py` was always loaded on CPU via `map_location="cpu"` and never moved to GPU, even when the main ViGNN model ran on CUDA. This caused CPU/GPU context switching overhead during the mixed-device prediction pipeline.

### Impact

| Metric | Before Fix (GPU host) | After Fix (GPU host) | CPU-only host |
|--------|----------------------|---------------------|---------------|
| Gate latency | **5,355ms** | **125ms** | 46ms |
| ViGNN inference | 270ms | 148ms | 87ms |
| Total pipeline | 5,625ms | **287ms** | 133ms |
| Speedup | — | **42x gate improvement** | — |

### Fix

Added auto-detection in `LearnedFundusGate.__init__()`:

```python
# Auto-detect device: use GPU if available, else CPU
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# After loading weights...
self.to(self._device)  # move model to detected device
```

And in `check()` / `check_tensor()`:

```python
tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self._device)
```

### Files Changed

- `src/data/fundus_gate_learned.py` — auto-device detection + tensor placement
- `OptiscanAI/src/data/fundus_gate_learned.py` — synced copy

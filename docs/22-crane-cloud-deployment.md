# 22. Crane Cloud Deployment Guide

Production deployment of OptiscanAI on [Crane Cloud](https://cranecloud.io) — Uganda's Kubernetes-as-a-Service platform.

## Docker Images

Two images are published to Docker Hub for different deployment targets:

| Tag | Base | Size | Use Case |
|-----|------|------|----------|
| `landwind/optiscan-ai:latest` | `nvidia/cuda:12.1.1-cudnn8-runtime` | ~9.8 GB | GPU servers with NVIDIA drivers |
| `landwind/optiscan-ai:cpu` | `python:3.11-slim-bookworm` | ~2.4 GB | Crane Cloud, CPU-only hosts |

Both images are full-stack: **nginx (port 8080) + FastAPI backend (8081) + Next.js frontend (3000)**, managed by supervisord.

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

## CPU Image (Crane Cloud)

### Why CPU?

- Crane Cloud's RENU cluster has no NVIDIA GPUs
- The AHUMAIN ML cluster (`supports_ml: true`) had scheduling issues during testing
- The CPU image is 2.4 GB vs 9.8 GB — pulls in under 60 seconds
- PyTorch CPU inference works for all 45 disease classes

### Model Weights Baked In

Crane Cloud has **no volume mount support**. Model weights must be inside the image:

```dockerfile
# In Dockerfile.cpu — weights baked in at build time
COPY models/model_vignn_rank1.pth ./models/model_vignn_rank1.pth   # 131 MB
COPY weights/fundus_gate.pth ./weights/fundus_gate.pth             # 6 MB
```

> **Important:** Do NOT rely on volume mounts (`-v`) or runtime model downloads on Crane Cloud.
> The platform's health checks will terminate the container before downloads complete.

### Deploying to Crane Cloud

#### Via Crane Cloud Web UI

1. Go to [cranecloud.io](https://cranecloud.io) and log in
2. Create a project (select **RENU** cluster — recommended)
3. Create an app with:
   - **Image:** `landwind/optiscan-ai:cpu`
   - **Port:** `8080`
   - **Replicas:** `1`
4. Set environment variables:

| Key | Value |
|-----|-------|
| `MODEL_PATH` | `models/model_vignn_rank1.pth` |
| `CUDA_VISIBLE_DEVICES` | `-1` |
| `DEVICE` | `cpu` |
| `FUNDUS_GATE__ENABLED` | `true` |

> **Warning:** Do not add leading/trailing spaces in env var keys or values.
> Crane Cloud passes them as-is, and the backend won't recognize ` DEVICE ` (with spaces).

#### Via Crane Cloud CLI

```bash
pip install cranecloud

# Login (use lowercase email)
cranecloud auth login

# If keyring errors occur on headless servers, use the API directly:
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

#### Via Crane Cloud API

```python
import requests

API = "https://api.cranecloud.io"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Create project on RENU cluster
requests.post(f"{API}/projects", headers=headers, json={
    "name": "OptiscanAI",
    "cluster_id": "9e81a70e-8460-4e5d-b0a8-17abcac30f68",  # RENU
    "owner_id": user_id
})

# Deploy app
requests.post(f"{API}/projects/{project_id}/apps", headers=headers, json={
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
```

### Crane Cloud Clusters

| Cluster | ID | Status | GPU | Recommendation |
|---------|-----|--------|-----|----------------|
| RENU | `9e81a70e-...` | Active | No | Use this for CPU deployments |
| AHUMAIN ML | `df2eeac2-...` | Active | `supports_ml: true` | Pod scheduling issues observed |
| Makerere-1 | `f3068db2-...` | Disabled | No | Not available |

### Verified Production URL

```
https://optiscan-ai-4fe6e1aa.renu-01.cranecloud.io
```

Endpoints tested and confirmed working:
- `/health` — `{"status": "healthy", "model_loaded": true, "device": "cpu", "diseases_count": 45}`
- `/` — Frontend UI (HTTP 200)
- `/docs` — Swagger API docs (HTTP 200)
- `/api/v1/gate/status` — Fundus Gate v2 enabled
- `/api/v1/predict` — 422 on non-fundus images (gate working)

## GPU Image (NVIDIA Servers / Crane Cloud AHUMAIN)

For GPU-enabled deployment (dedicated servers, cloud VMs with NVIDIA drivers, or Crane Cloud AHUMAIN ML cluster):

```bash
docker pull landwind/optiscan-ai:latest
```

### Running Locally with GPU

```bash
docker run -d \
  --gpus '"device=0"' \
  -p 8080:8080 \
  -v ./models:/app/models \
  -v ./weights:/app/weights \
  landwind/optiscan-ai:latest
```

### Crane Cloud AHUMAIN (GPU)

Deploy via API with GPU configuration:

```python
requests.post(f"{API}/projects/{project_id}/apps", headers=headers, json={
    "name": "optiscan-ai-gpu",
    "image": "landwind/optiscan-ai:latest",
    "port": 8080,
    "replicas": 1,
    "env_vars": {
        "MODEL_PATH": "models/model_vignn_rank1.pth",
        "CUDA_VISIBLE_DEVICES": "0",
        "DEVICE": "cuda",
        "FUNDUS_GATE__ENABLED": "true"
    }
})
```

> **Note:** The GPU image does NOT bake model weights — it expects volume mounts.
> If AHUMAIN supports persistent volumes, mount them. Otherwise, use the CPU image.

## Building Images

```bash
# CPU image (Crane Cloud)
make docker-build-cpu
# or: docker build -t landwind/optiscan-ai:cpu -f Dockerfile.cpu .

# GPU image (NVIDIA servers)
make docker-build
# or: docker build -t landwind/optiscan-ai:latest -f Dockerfile .

# Push to Docker Hub
docker push landwind/optiscan-ai:cpu
docker push landwind/optiscan-ai:latest
```

## CI/CD Pipeline

The GitHub Actions workflow `.github/workflows/docker-publish.yml` automates:

1. **Trigger:** Push to `main` (Dockerfile/src/backend/configs changes), version tags (`v*`), manual dispatch
2. **Test gate:** Lint (ruff + black) + pytest
3. **Build & push:** GPU image to `landwind/optiscan-ai:latest`

To add the CPU image to CI, extend the matrix:

```yaml
strategy:
  matrix:
    include:
      - variant: gpu
        dockerfile: Dockerfile
        image: optiscan-ai
      - variant: cpu
        dockerfile: Dockerfile.cpu
        image: optiscan-ai
        tag: cpu
```

## Docker Best Practices Applied (26/26)

Both Dockerfiles follow all industry standards:

- 3-stage multi-stage build (Python builder, frontend builder, slim runtime)
- `SHELL ["/bin/bash", "-euo", "pipefail", "-c"]` for error handling
- OCI image-spec labels (`org.opencontainers.image.*`)
- Non-root user (`optiscan`)
- `STOPSIGNAL SIGTERM` with graceful shutdown per process
- Secure Node.js install via GPG key (no `curl | bash`)
- `--no-install-recommends` on all apt-get
- `PIP_NO_CACHE_DIR=1` and `--no-cache-dir`
- Pinned dependency versions
- `.dockerignore` (build context ~20 KB)
- `HEALTHCHECK` with 60s start-period
- Stripped streamlit, pyarrow, opencv-python (headless only)
- Cleaned `__pycache__`, `.dist-info`, test dirs from venv

## Troubleshooting

### Pod status "unknown" on Crane Cloud

This indicates a cluster scheduling issue, not an image problem. Steps:

1. Delete the app and recreate it
2. Try a different cluster (RENU vs AHUMAIN)
3. Check that env var keys have no leading/trailing spaces

### Container exits immediately

The model weights are likely missing. Ensure:
- CPU image: weights are baked in (check `docker run --rm landwind/optiscan-ai:cpu ls /app/models/`)
- GPU image: volumes are mounted (`-v ./models:/app/models`)

### Health check timeout

The app takes ~30-40s to start on CPU (model loading + Next.js boot). The HEALTHCHECK has a 60s `start-period` to account for this. If Crane Cloud terminates earlier, the app needs a longer grace period.

### Env vars with spaces

Crane Cloud passes env vars exactly as entered. ` DEVICE ` (with spaces) is NOT the same as `DEVICE`. Always verify via the API:

```python
resp = requests.get(f"{API}/apps/{app_id}", headers=headers)
print(resp.json()["data"]["apps"]["env_vars"])
```

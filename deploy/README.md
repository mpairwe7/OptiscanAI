# `deploy/` — deployment & build files

Container build definitions, Compose stacks, and the in-container reverse
proxy / process manager configs. Grouped here to keep the repo root clean.

> **Build context is always the repository root.** Compose files and the CI
> `docker build` commands pass the Dockerfile with `-f deploy/<file>` while the
> context stays `.` (or `..` inside the Compose files), so every `COPY` path in
> the Dockerfiles resolves against the repo root, not this directory.

## Contents

| File | Purpose |
| --- | --- |
| `Dockerfile` | Production GPU image (CUDA 12.4, full stack via supervisord) |
| `Dockerfile.cpu` | CPU-only image (Crane Cloud, generic clouds) |
| `Dockerfile.hf` | Hugging Face Spaces image (port 7860, CPU, embedded Postgres) |
| `docker-compose.yml` | Base stack: Postgres sidecar + GPU `api` (+ `cpu`/`hf` profiles); pins `name: optiscan` |
| `docker-compose.otel.yml` | Overlay — OpenTelemetry Collector + Jaeger + Prometheus |
| `docker-compose.mlflow.yml` | Overlay — MLflow tracking server |
| `docker-compose.2026.yml` | Overlay — full stack (Ray Serve, Kafka, observability) |
| `nginx.conf` | Reverse proxy (nginx → uvicorn + Next.js); used by `Dockerfile.hf` |
| `supervisord.conf` | Process manager (backend + frontend + nginx); used by `Dockerfile.hf` |

## Usage (run from the repo root)

```bash
# Build
docker build -f deploy/Dockerfile     -t optiscan-ai:latest .   # GPU
docker build -f deploy/Dockerfile.cpu -t optiscan-ai:cpu    .   # CPU

# Compose (single-file or with overlays)
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml --profile cpu up -d
docker compose -f deploy/docker-compose.yml --profile hf  up --build

# Phase stacks (see Makefile: make up-phase1 / up-phase2 / up-full)
docker compose -f deploy/docker-compose.yml \
               -f deploy/docker-compose.otel.yml \
               -f deploy/docker-compose.mlflow.yml up -d
```

The `Makefile` targets (`make docker-build`, `make up-phase1`, `make hf-local`,
…) already use these `deploy/` paths. See `docs/20-deployment-rollout-guide.md`
and `docs/22-crane-cloud-deployment.md` for full deployment runbooks.

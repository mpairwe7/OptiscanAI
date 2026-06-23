# Troubleshooting

Common failures and fixes. See also [`docs/06-backend-setup.md`](06-backend-setup.md),
[`docs/22-crane-cloud-deployment.md`](22-crane-cloud-deployment.md), and the
docs index [`docs/README.md`](README.md).

## Model loading

| Symptom | Cause | Fix |
| --- | --- | --- |
| `_pickle.UnpicklingError: invalid load key, 'v'` or `/health` → `model_loaded:false` | The checkpoint is a **Git LFS pointer**, not the real weights | `git lfs pull` (CI does this before tests/builds). Confirm `stat -c%s models/model_vignn_rank1.pth` is multi-MB. |
| `CUDA out of memory` | GPU too small / batch too large | Lower batch size in `configs/*.yaml`; set `CUDA_VISIBLE_DEVICES=-1` / `DEVICE=cpu` to force CPU; use a quantized format. |
| Health check never goes ready (slow startup) | Model load > healthcheck `start-period` | Increase `start-period` in the Dockerfile / Compose healthcheck, or use a smaller checkpoint. |

## Containers & deployment

| Symptom | Cause | Fix |
| --- | --- | --- |
| `docker compose` can't find the file | Compose files live in `deploy/` | Run from the repo root with `-f deploy/docker-compose.yml` (the Makefile targets already do). |
| Supervisor crash on HF Spaces (`%(ENV_CUDA_VISIBLE_DEVICES)s` expansion fails) | HF doesn't inject those env vars | They're hardcoded (`CUDA_VISIBLE_DEVICES=-1`, `DEVICE=cpu`) in `deploy/supervisord.conf`. |
| `COPY` fails in `deploy/Dockerfile.hf` | Build run with the wrong context | Build context must be the **repo root** (`docker build -f deploy/Dockerfile.hf .`); HF/`deploy_hf.sh` rsync `deploy/` into the Space first. |
| Port already in use (8080/3000/7860) | Another process bound the port | Stop it or remap (`API_PORT`, Compose `ports:`). |

## Deployment — Crane Cloud

| Symptom | Cause | Fix |
| --- | --- | --- |
| `deploy-crane-cloud` fails with `HTTP 500` and a body like `HTTPSConnectionPool(host='hub.docker.com', ...): ... Temporary failure in name resolution` | Crane Cloud's backend validates the image tag by calling Docker Hub (`hub.docker.com`); **its own server can't resolve DNS / reach Docker Hub** — an upstream Crane infra issue, not a repo bug. The image exists; Crane just can't reach Docker Hub to verify it. | Wait for Crane Cloud to recover, then re-roll **without a rebuild**: `gh workflow run docker-publish.yml -f deploy_only=true -f redeploy_sha=<7-char-sha>`. Confirm the API is back first — an env-only run (`-f deploy_only=true`, no `redeploy_sha`) returns `HTTP 200` even while image rollouts 500. If it stays broken, report to Crane Cloud support. |
| `deploy-crane-cloud` shows only `HTTP 500 Internal Server Error`, no detail | Old `crane_deploy.py` swallowed the response body | Fixed: the script now logs Crane's body (DSN/password redacted) and retries 5xx up to 3×. |
| Need to roll back or re-attempt a deploy | App is on a bad image, or a deploy failed after a good build | `gh workflow run docker-publish.yml -f deploy_only=true -f redeploy_sha=<sha>` re-rolls an already-built image (no rebuild); omitting `redeploy_sha` just re-asserts the managed-DB env. |

## Database (Phase 6)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Billing endpoints 5xx after deploy | `alembic upgrade` failed at startup | Check logs; run `alembic -c backend/alembic.ini upgrade head` manually. See [`docs/26`](26-database-migrations.md). |
| `asyncpg` auth/connection error | Wrong `DATABASE__URL` / unencoded password | Use `postgresql+asyncpg://`, **percent-encode** special chars, verify `POSTGRES_HOST/PORT/USER` mirror the URL. |
| Connecting to managed DB hangs | `EMBEDDED_POSTGRES__ENABLED` still on, or TLS required | Set `EMBEDDED_POSTGRES__ENABLED=false`; append `?ssl=require`. |

## Lint / build / tooling

| Symptom | Cause | Fix |
| --- | --- | --- |
| CI `black --check` fails but it's formatted | Local black ≠ pinned version | Use `black==26.3.1` (matches CI + `pyproject.toml`). |
| `uv lock` fails to resolve with the `federated` extra | black 26.x `pathspec≥1.0` vs flwr `pathspec<0.13` | Expected — they're declared mutually exclusive via `[tool.uv].conflicts`; don't install `dev` + `federated` together. |
| `ModuleNotFoundError: src` / `backend` | Missing `PYTHONPATH` | Prefix commands with `PYTHONPATH=.` (Makefile targets do). |
| Frontend build / postcss issues | Stale lockfile | `cd frontend && bun install`; postcss is pinned ≥ 8.5.15 via `overrides`. |

## Secrets

Never paste secrets into issues, logs, or commits. If a key leaks, rotate it and
scrub history. See [`SECURITY.md`](../SECURITY.md).

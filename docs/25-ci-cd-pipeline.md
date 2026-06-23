# CI/CD Pipeline

All automation lives in [`.github/workflows/`](../.github/workflows/). Six
workflows cover linting/testing, image build & deploy, security, HF Spaces, and
the SaaS smoke + quantization gates.

| Workflow | Triggers | What it does |
| --- | --- | --- |
| `ml-pipeline.yml` | push/PR to `main`/`dev`, manual | **Lint** (ruff + `black --check`, pinned `black==26.3.1`) then **test** (`pytest`, LFS-smudged for model load). The primary quality gate. |
| `docker-publish.yml` | push to `main`/`dev` (on `deploy/Dockerfile*`, `src/`, `backend/`, `frontend/`, `configs/`, `models/`, …), tags `v*`, manual | Test → build & push GPU + CPU images (matrix `dockerfile: deploy/Dockerfile{,.cpu}`, context = repo root) to Docker Hub → deploy to Crane Cloud. |
| `security-scan.yml` | push/PR, weekly (Mon 06:00) | `pip-audit` (`requirements.txt`), Trivy image scan (builds `deploy/Dockerfile.cpu`) → SARIF, TruffleHog secrets, SPDX SBOM artifact. |
| `deploy-hf-spaces.yml` | push to `main`/`dev` (on `src/`, `backend/`, `frontend/`, `deploy/Dockerfile.hf`, …), manual | Test, then sync the repo into the HF Space (`cp deploy/Dockerfile.hf → Dockerfile`) and push to `Mpairwe49/retinal-screening`. |
| `billing-smoke.yml` | push/PR | Smoke tests for the subscription/billing platform. |
| `quantization.yml` | push/PR | Quantization quality gate (faithfulness drop + p95 latency thresholds). |

## Required secrets

Set these as GitHub repo/org secrets (Settings → Secrets and variables → Actions):

| Secret | Used by |
| --- | --- |
| `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | `docker-publish` |
| `CRANE_CLOUD_EMAIL`, `CRANE_CLOUD_PASSWORD`, `CRANE_CLOUD_CPU_APP_ID`, `CRANE_CLOUD_GPU_APP_ID`, `CRANE_CLOUD_CPU_URL`, `DATABASE_URL` | `docker-publish` (deploy stage) |
| `HF_TOKEN`, `HF_USER` | `deploy-hf-spaces` |

## Local equivalents

Run the same gates before pushing:

```bash
ruff check src/ backend/ scripts/ train.py
black --check src/ backend/ scripts/ train.py     # black==26.3.1
make test                                          # pytest
make docker-build && make docker-build-cpu         # images (deploy/Dockerfile*)
```

## Notes

- Editing files under `.github/workflows/` over HTTPS requires the pushing
  token to carry the `workflow` OAuth scope.
- Model weights are Git LFS; CI runs `git lfs pull` before tests/builds so
  `torch.load()` gets real checkpoints (not LFS pointers).
- `black` is pinned identically in CI and `pyproject.toml`; bump them together.

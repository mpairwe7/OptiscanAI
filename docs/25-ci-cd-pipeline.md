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
| `GEMINI_API_KEY` | both — agentic AI. Unset ⇒ the agent graph serves deterministic clinical rules |
| `SUNBIRD__API_TOKEN`, `SUNBIRD__FALLBACK_API_TOKEN` | both — Sunbird cloud voice. Unset ⇒ voice stays local-only (whisper/piper) |

Optional repo **variables** (same screen, Variables tab):

| Variable | Effect |
| --- | --- |
| `GEMINI_MODEL` | Model pin. Unset ⇒ `gemini-3.7-flash` |
| `SUNBIRD__ENABLED` | `false` stages the rollout. Unset ⇒ `true` whenever a Sunbird token is present |

Both providers are pushed to Crane Cloud (`crane_deploy.py`) and set as Hugging
Face Space secrets (Hub API) by their deploy workflows, and each is inert when
its secret is absent — so a missing one degrades that feature rather than
failing the deploy. The `SUNBIRD__*` double underscore is pydantic-settings'
nested delimiter (it populates `settings.sunbird.*`) and must be preserved in
the secret name.

> **Sunbird failover needs two accounts.** With only `SUNBIRD__API_TOKEN` set,
> a daily-quota `429` silently drops Ugandan narration back to an English voice
> with nothing to fail over to, while the tier still reports itself available.
> Both deploy paths emit a CI warning in that case.

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
- Crane Cloud's `PATCH /apps/{id}` `env_vars` is **add-only** — it adds keys but
  never overwrites one that exists. So a provider token (or `SUNBIRD__ENABLED`)
  that has already landed cannot be changed from CI; rotate it in the Crane
  console. Hugging Face Space secrets *do* overwrite, so CI rotates those.
- Model weights are Git LFS; CI runs `git lfs pull` before tests/builds so
  `torch.load()` gets real checkpoints (not LFS pointers).
- `black` is pinned identically in CI and `pyproject.toml`; bump them together.

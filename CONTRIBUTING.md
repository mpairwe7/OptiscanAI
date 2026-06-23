# Contributing to OptiscanAI

Thanks for your interest in improving **OptiscanAI** — a multi-label retinal
disease screening platform (ML training + FastAPI backend + Next.js frontend).
This guide covers local setup, conventions, and the contribution workflow.

## Prerequisites

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/) ≥ 1.0 (frontend)
- Git LFS (model weights are LFS-tracked)
- Docker (optional — container/compose work lives in [`deploy/`](deploy/README.md))

## Local setup

```bash
# Backend / ML  (resolves the env from pyproject.toml; uv.lock is gitignored)
uv sync

# Frontend
cd frontend && bun install && cd ..

# Run both — backend :8080, frontend :3000
make dev
```

The project runs via `PYTHONPATH=.` (no editable install). Common entry points:

```bash
make backend        # uvicorn backend.app.main:app --reload  (:8080)
make frontend       # next dev                                 (:3000)
PYTHONPATH=. python train.py --config configs/train.yaml
```

See the docs index ([`docs/README.md`](docs/README.md)) — especially
`06-backend-setup.md`, `07-frontend-setup.md`, and `24-environment-variables.md`.

## Code style & quality gates

Python is linted with **ruff** and formatted with **black**, pinned to
`black==26.3.1` (CI checks this exact version — do not reformat with another
black release). TypeScript is `strict` mode + ESLint.

```bash
ruff check src/ backend/ scripts/ train.py
black --check src/ backend/ scripts/ train.py
cd frontend && bun run lint
```

- Type everything (Python hints, TS strict); no new untyped public surfaces.
- Match the surrounding code's conventions; don't add a second way to do
  something that already exists.
- No dead code, commented-out blocks, or TODOs without a tracking issue.

## Tests

```bash
make test            # PYTHONPATH=. pytest tests/ -v
make test-fast       # stop on first failure
PYTHONPATH=. pytest -m "not slow"   # skip slow-marked tests
```

Add or extend tests for new logic. CI (`.github/workflows/ml-pipeline.yml`) runs
ruff + black + pytest and must be green before merge.

## Commits & pull requests

- Branch off `dev`; open PRs into `dev` (releases flow `dev → main`).
- Use [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, `build:`, `ci:` …). Do **not** add
  `Co-Authored-By` trailers.
- Keep PRs small and focused; fill in the PR template; get CI green.
- Never commit secrets or `.env*` files — see [`SECURITY.md`](SECURITY.md).

## Reporting bugs / requesting features

Use the GitHub issue templates (bug report / feature request). For security
issues, follow [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.

By contributing, you agree your contributions are licensed under the repository
[`LICENSE`](LICENSE) (CC-BY-4.0).

# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Force `postcss` ≥ 8.5.15 via an npm/bun `overrides` entry to patch
  **GHSA-qx2v-qp2m-jg93** (CSS-stringify XSS) that Next.js pulled in
  transitively at `postcss@8.4.31`.
- Bump `black` 25.12.0 → 26.3.1 to patch **GHSA-3936-cmfr-pm3m** (arbitrary
  file write via cache filename). black 26.x needs `pathspec≥1.0`, which is
  incompatible with flwr's `pathspec<0.13`, so the dev group/extra is declared
  mutually exclusive with the `federated` extra via `[tool.uv].conflicts`.
- Force `@babel/core` ≥ 7.29.6 via the frontend `overrides` to patch
  **GHSA-4x5r-pxfx-6jf8 / CVE-2026-49356** (arbitrary file read via
  `sourceMappingURL`), which Next.js pulled in transitively at `7.29.0`
  (Dependabot #17). Both lockfiles now resolve `@babel/core@7.29.7`.

### Added
- **Sunbird AI cloud tier for Ugandan-language voice** (`SUNBIRD__*`, disabled
  by default). `backend/app/core/sunbird_client.py` adds ASR, TTS and
  translation for Luganda, Runyankole, Acholi, Swahili, Ateso and Lugbara, with
  dual-account failover and 429/5xx retry. Wired *behind* the local models:
  whisper/piper run first and the cloud is consulted only when they return
  nothing, so an offline clinic is unaffected. Partial transcriptions never
  leave the device — a round-trip per 500ms chunk would destroy the streaming
  latency the partial exists for.
- **Navigation rail** — the desktop sidebar collapses to an icon rail and
  expands on hover, with a pin control (`Keep open`) persisted in
  `localStorage`. A pre-paint inline script stamps `data-rail-mode` on `<html>`
  so a pinned rail is full width in the first painted frame rather than
  animating open after hydration; width and label visibility are CSS state, not
  React state. Replaces the unpersisted `sidebarCollapsed` store flag.
- **Google Gemini is now the agentic-AI provider** (`google-genai`, model pin
  `gemini-3.7-flash` via `GEMINI_MODEL`). The LLM layer gained a client-side
  requests-per-minute throttle (`GEMINI_RPM`, free-tier default 10) and a
  reasoning-token budget (`GEMINI_THINKING_HEADROOM` / `GEMINI_MIN_OUTPUT_TOKENS`).
- CI now provisions the provider key: `crane_deploy.py` PATCHes `GEMINI_API_KEY`
  onto the Crane apps, and the HF Spaces workflow sets it as a Space secret via
  the Hub API. Both are inert when the `GEMINI_API_KEY` repo secret is unset.
- `requirements.txt` — pip mirror of the core dependencies so plain-pip
  workflows (and the `pip-audit` security scan) work without uv.
- Crane Cloud deploy diagnostics & resilience (`.github/scripts/crane_deploy.py`):
  the deploy step now logs Crane's HTTP error **body** (with the DB DSN/password
  redacted) instead of a bare `HTTP 500`, and retries `5xx` up to 3× with linear
  backoff. A new `redeploy_sha` `workflow_dispatch` input lets `deploy_only` runs
  re-roll an already-built image without a rebuild (rollback / deploy re-attempt).
- Repo-hygiene files: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CITATION.cff`, `.editorconfig`, PR + issue templates, and
  `.github/dependabot.yml`.
- Documentation: `docs/README.md` index, `docs/24-environment-variables.md`,
  `docs/25-ci-cd-pipeline.md`, `docs/26-database-migrations.md`, and
  `docs/27-troubleshooting.md`; project-specific `frontend/README.md`.

### Removed
- **Anthropic Claude and Groq decommissioned.** `ANTHROPIC_API_KEY`,
  `ANTHROPIC_ORG_ID`, `AGENT_MODEL`, and all `GROQ_*` settings are gone; the
  fallback chain is now Gemini → deterministic rules (was Claude → Groq →
  rules). The `claude_agent` / `groq_agent` circuit breakers collapse into a
  single `gemini_agent`, and the k8s secret key `anthropic-api-key` becomes
  `gemini-api-key`.

### Fixed
- **The agent graph never reached any LLM in production.** `src/agents/llm.py`
  read provider keys from `os.environ`, but pydantic-settings loads `.env` into
  the `Settings` object without exporting it to the process environment — so a
  key set in `.env` was invisible and `/api/v1/agents/graph/info` reported
  `active_provider: "none"` on both Hugging Face and Crane Cloud. The layer now
  reads through `Settings`. No LLM SDK was declared in `pyproject.toml` either,
  so the import would have failed regardless.
- Truncated LLM output can no longer reach a clinical report. Gemini 3.x charges
  thinking tokens against `max_output_tokens` and ignores `thinking_budget=0`,
  so the graph's 200/300-token call sites returned mid-sentence fragments with
  `finish_reason=MAX_TOKENS`. Those are now rejected as a fallback rather than
  passed through as the narrative.
- `gemini_api_key` is a Pydantic `SecretStr`, so `print(settings)` and
  structured-log dumps render `**********` instead of the key.

### Changed
- Group deployment/build files under `deploy/` (the three Dockerfiles, the four
  docker-compose overlays, `nginx.conf`, `supervisord.conf`). References updated
  across the Makefile, CI workflows, scripts, and docs; Compose pins
  `name: optiscan`. Build context remains the repo root.
- Move `improvement_report.md` into `docs/`.

[Unreleased]: https://github.com/mpairwe7/OptiscanAI/compare/main...dev

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

### Added
- `requirements.txt` — pip mirror of the core dependencies so plain-pip
  workflows (and the `pip-audit` security scan) work without uv.
- Repo-hygiene files: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CITATION.cff`, `.editorconfig`, PR + issue templates, and
  `.github/dependabot.yml`.
- Documentation: `docs/README.md` index, `docs/24-environment-variables.md`,
  `docs/25-ci-cd-pipeline.md`, `docs/26-database-migrations.md`, and
  `docs/27-troubleshooting.md`; project-specific `frontend/README.md`.

### Changed
- Group deployment/build files under `deploy/` (the three Dockerfiles, the four
  docker-compose overlays, `nginx.conf`, `supervisord.conf`). References updated
  across the Makefile, CI workflows, scripts, and docs; Compose pins
  `name: optiscan`. Build context remains the repo root.
- Move `improvement_report.md` into `docs/`.

[Unreleased]: https://github.com/mpairwe7/OptiscanAI/compare/main...dev

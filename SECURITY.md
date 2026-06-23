# Security Policy

OptiscanAI processes clinical images and runs a subscription/billing platform,
so security and privacy are first-class concerns.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately via GitHub
[Security Advisories](https://github.com/mpairwe7/OptiscanAI/security/advisories/new)
("Report a vulnerability"), or contact the maintainer listed on the
[`mpairwe7`](https://github.com/mpairwe7) GitHub profile. Please include:

- a description of the issue and its impact,
- steps to reproduce (a proof of concept if possible),
- the affected version / commit.

We aim to acknowledge reports within 5 business days and to share a remediation
timeline after triage.

## Supported versions

Security fixes target the `main` branch and the latest published image tags.
Older tags are not patched.

| Version | Supported |
| --- | --- |
| `main` / latest images | ✅ |
| older tags | ❌ |

## Automated scanning

Every push and a weekly schedule run
[`.github/workflows/security-scan.yml`](.github/workflows/security-scan.yml):

- **pip-audit** — Python dependency CVEs (`requirements.txt`)
- **Trivy** — container image scan (CRITICAL/HIGH) → GitHub code scanning
- **TruffleHog** — verified secret detection
- **SBOM** — SPDX SBOM artifact (anchore/sbom-action)

Dependency updates are proposed automatically by Dependabot
([`.github/dependabot.yml`](.github/dependabot.yml)).

## Handling secrets

Never commit secrets or `.env*` files. Configure via environment variables /
secrets managers (see [`docs/24-environment-variables.md`](docs/24-environment-variables.md)).
Provider keys and tokens must be sent in request **headers** (never URL query
strings), stored as `SecretStr`/env, and never logged.

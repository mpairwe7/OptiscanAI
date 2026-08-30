# OptiscanAI Documentation

Index of all runbooks and reference docs. Start with **Getting Started**, then
dive into the area you need. (Repo overview lives in the root
[`README.md`](../README.md).)

## Getting Started

| Doc | What it covers |
| --- | --- |
| [06 — Backend Setup](06-backend-setup.md) | FastAPI app, routers, middleware, auth, config, running locally + in Docker |
| [07 — Frontend Setup](07-frontend-setup.md) | Next.js 16 + Bun + Zustand + TanStack Query; structure, run, build, deploy modes |
| [24 — Environment Variables](24-environment-variables.md) | Every env var (backend nested settings + frontend `NEXT_PUBLIC_*`) |
| [26 — Database & Migrations](26-database-migrations.md) | PostgreSQL + Alembic workflow (async) |
| [27 — Troubleshooting](27-troubleshooting.md) | Common failures and fixes |

## Architecture & Model

| Doc | What it covers |
| --- | --- |
| [Architecture — RetinalFoundationHybrid](architecture-retinal-foundation-hybrid.md) | System + model architecture (RETFound + LoRA + graph head + uncertainty) |
| [03 — Training](03-training.md) | Model architectures, DDP, LoRA, losses |
| [15 — Image Gating](15-image-gating.md) · [16 — Fundus Gate v2](16-fundus-gate-v2.md) | Fundus-only inference protection (statistical + learned fusion) |

## ML Pipeline (Data → Train → Eval → Export)

| Doc | What it covers |
| --- | --- |
| [01 — Data Ingestion](01-data-ingestion.md) · [02 — Data Augmentation](02-data-augmentation.md) | Dataset, RFMiD, augmentation pipeline |
| [04 — Evaluation](04-evaluation.md) · [precision-rescue-verification](precision-rescue-verification.md) | Metrics, calibration, precision-rescue |
| [28 — Reasoner: CNN vs DistilledQwen](28-reasoner-cnn-vs-distilledqwen.md) | Self-contained replacement for the external LLM reasoner: design, harness, feasibility, Go/No-Go |
| [29 — Narrator verification & gaps](29-narrator-verification-and-gaps.md) | Review of doc 28 §0.6–§0.9: what verified, what was wrong (truncation confound, non-comparative latency, grounding blind spots), and the gap roadmap |
| [05 — Model Export](05-model-export.md) | ONNX / TorchScript / INT8 export to backend |
| [12 — Advanced MLOps](12-advanced-mlops.md) | DVC pipeline, MLflow, HPO, retraining |

## MLOps, Observability & Rollout

| Doc | What it covers |
| --- | --- |
| [08 — Production Improvements](08-production-improvements.md) | Monitoring, drift, active learning, HF hardening |
| [17 — Implementation Roadmap](17-implementation-roadmap.md) · [18 — Migration Guide](18-migration-guide.md) | Phase-by-phase activation |
| [25 — CI/CD Pipeline](25-ci-cd-pipeline.md) | The 6 GitHub Actions workflows + secrets |
| [09 — Testing](09-testing.md) | Test suite, fixtures, quality gates |

## Deployment

| Doc | What it covers |
| --- | --- |
| [20 — Deployment & Rollout Guide](20-deployment-rollout-guide.md) | Phased rollout, Docker/K8s |
| [22 — Crane Cloud Deployment](22-crane-cloud-deployment.md) | Uganda K8s, 3 Postgres shapes, registry |
| [21 — vLLM GPU7 AWQ Optimization](21-vllm-gpu7-awq-optimization.md) | GPU/runtime tuning |
| [19 — Offline / Mobile Optimization](19-offline-mobile-optimization.md) | Offline-first, voice-first, quantization |
| [`deploy/README.md`](../deploy/README.md) | Dockerfiles + Compose overlays + proxy/process configs |

## Governance, Security & Compliance

| Doc | What it covers |
| --- | --- |
| [10 — Security](10-security.md) | Auth, rate limiting, scanning, SBOM |
| [11 — Governance](11-governance.md) | Model/dataset cards, fairness, audit trail |
| [iso14971 Risk Analysis](iso14971_risk_analysis.md) | Medical-device risk management |
| [`SECURITY.md`](../SECURITY.md) | Vulnerability disclosure policy |

## Business & Product

| Doc | What it covers |
| --- | --- |
| [23 — Billing Platform](23-billing-platform.md) | SaaS tiers, identity/tenancy, payment rails |
| [13 — Commercialization Strategy](13-commercialization-strategy.md) · [14 — Hackathon & Pitch Guide](14-hackathon-pitch-guide.md) | GTM, market, pitch |
| [Bill of Materials](Bill-of-Materials.md) · [IEEE Final Report](IEEE_Final_Report.md) · [improvement-report](improvement-report.md) | BOM, research paper, audit report |

---

PDFs (pitch guide, commercialization, video script) and `convert_to_pdf.py` also
live in this directory. Contributing? See [`CONTRIBUTING.md`](../CONTRIBUTING.md).

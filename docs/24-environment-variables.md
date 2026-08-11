# Environment Variables

Centralized reference for every environment variable. The authoritative sources
are [`.env.example`](../.env.example) (copy it to `.env`) and the
pydantic-settings classes in `backend/app/core/config.py`.

## How configuration is loaded

- The backend reads `.env` (local) and process environment via **pydantic-settings**.
- **Nested settings** use a double-underscore delimiter: `SECTION__FIELD`.
  Example: `DATABASE__URL`, `FUNDUS_GATE__ENABLED`, `STRIPE__API_KEY`.
- Precedence: process env > `.env` file > defaults in `config.py`.
- Most feature blocks are **opt-in** (`*__ENABLED=false` by default). Enabling a
  block is what activates the corresponding Phase (see `docs/17`, `docs/18`).
- **Never commit `.env`** or real secret values. Send provider keys in request
  headers, never URL query strings. See [`SECURITY.md`](../SECURITY.md).

## Core

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_PATH` | `models/model_vignn_rank1.pth` | Active checkpoint |
| `MODEL_NAME` | `vignn` | Architecture key |
| `TRIAGE_MODEL_ENABLED` | `true` | Use the local learned triage head instead of calling an LLM. `false` restores the LLM → rules path |
| `TRIAGE_MODEL_PATH` | `models/triage/triage_model.json` | 3 KB JSON weights for the triage head. Missing/unreadable ⇒ falls back to rules (never fails a scan) |
| `NUM_CLASSES` | `45` | Multi-label output size |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8080` | Uvicorn bind |
| `DEBUG` | `false` | |
| `CUDA_VISIBLE_DEVICES` | `0` | `-1` for CPU-only |
| `DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `CORS_ORIGINS` | (unset) | JSON list of allowed origins |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | `json` \| `text` |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP token bucket |
| `PREDICTION_LOG_DIR` | `logs/predictions` | |
| `MAX_UPLOAD_SIZE` | `10485760` | Bytes (10 MB) |

## Authentication

| Variable | Default | Notes |
| --- | --- | --- |
| `AUTH_ENABLED` | `false` | **Set `true` in production** |
| `JWT_SECRET` | _(placeholder)_ | **Required** in prod (validator enforces). Generate: `openssl rand -hex 32` |
| `JWT_ALGORITHM` | `HS256` | |
| `JWT_ACCESS_TTL_SECONDS` | `900` | Access token TTL |
| `JWT_REFRESH_TTL_SECONDS` | `2592000` | Refresh token TTL (30 d) |

## Agentic AI (LLM fallback chain: Claude → Groq → deterministic)

| Variable | Notes |
| --- | --- |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_ORG_ID` | Primary (Claude). Default `AGENT_MODEL=claude-sonnet-4-20250514` |
| `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_MAX_TOKENS`, `GROQ_TEMPERATURE` | Fallback |
| `AGENT_MONITOR_INTERVAL`, `AGENT_GOVERNANCE_INTERVAL` | Scheduling (seconds) |

> Keep provider keys as secrets/env only — never log them or place them in URLs.

## Fundus Gate v2 (`FUNDUS_GATE__*`)

`ENABLED`, `VERSION` (`v1`\|`v2`), `LEARNED_WEIGHT` (0.4), `MIN_CONFIDENCE`
(0.70), `MODEL_PATH` (`weights/fundus_gate.pth`), `VISUAL_EVIDENCE`.

## Phase 1 — Observability & MLOps

| Block | Key vars |
| --- | --- |
| `TELEMETRY__*` | `ENABLED`, `OTLP_ENDPOINT`, `SERVICE_NAME`, `SAMPLE_RATE` |
| `MLFLOW__*` | `ENABLED`, `TRACKING_URI`, `MODEL_NAME`, `EXPERIMENT_NAME` |
| `ACTIVE_LEARNING_LOOP__*` | `ENABLED`, `RETRAIN_THRESHOLD`, `QUEUE_DIR` |
| `DRIFT__*` | `ENABLED`, `CHECK_INTERVAL`, `NANNYML_ENABLED`, `EVIDENTLY_ENABLED`, `ALERT_WEBHOOK_URL` |

## Phase 2/3/4 — Scale, Edge, Governance, Future

`RAY__*`, `KAFKA__*`, `MTLS__ENABLED`, `EDGE__{ONNX,COREML,QUANTIZED}_ENABLED`,
`FAIRNESS__ENABLED`, `RESILIENCE__ENABLED`, `MULTIMODAL__ENABLED`,
`FEDERATED__ENABLED`. All boolean opt-in; see `.env.example` for endpoints.

## Phase 5 — Offline / Mobile / Voice

| Block | Key vars |
| --- | --- |
| `OFFLINE_RAG__*` | `ENABLED`, `INDEX_DIR`, `SOURCE_DIR`, `BUNDLES_DIR`, `EMBEDDER_PATH`, `TOP_K`, `SIMILARITY_THRESHOLD`, `SYNC_INTERVAL_S`, `COMPRESSION`, `TARGET_BUNDLE_SIZE_MB` |
| `QUANTIZATION__*` | `ENABLED`, `ACTIVE_FORMAT` (`gguf_q4_k_m`\|`awq_4bit`\|`onnx_int8`), `TORCH_COMPILE_*`, `VLLM_ENABLED`, `MAX_FAITHFULNESS_DROP`, `MAX_P95_LATENCY_MS`, … |
| `VOICE_FIRST__*` | `ENABLED`, `DEFAULT_LANGUAGE` (`en-ug`), `ASR_MODEL(_PATH)`, `TTS_ENGINE`/`TTS_MODEL_PATH`, `VAD_SENSITIVITY`, `BARGE_IN_ENABLED`, `SPEECH_RATE` |
| `MOBILE_BUNDLE__*` | `ENABLED`, `MAX_BUNDLE_SIZE_MB`, `TARGET_MODEL`, `INCLUDE_VOICE_MODELS`, `MIN_ANDROID_SDK`, `MIN_RAM_MB` |

## Phase 6 — Subscription billing (SaaS layer)

See [`docs/23-billing-platform.md`](23-billing-platform.md). Activated by flipping
the blocks below on.

| Block | Key vars |
| --- | --- |
| `DATABASE__*` | `ENABLED`, `URL` (`postgresql+asyncpg://…`), `POOL_SIZE`, `MAX_OVERFLOW`, `ECHO` |
| Postgres (sidecar) | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`, `EMBEDDED_POSTGRES__ENABLED` |
| `BILLING__*` | `ENABLED`, `FREE_SCAN_LIMIT_MONTHLY`, `FREE_PERIOD_DAYS`, `QUOTA_CACHE_TTL_S`, `ANNUAL_DISCOUNT_PCT` |
| `EMAIL__*` | `ENABLED`, `PROVIDER` (`console`\|`smtp`\|`resend`\|`sendgrid`), `FROM_*`, provider keys/SMTP |
| `STRIPE__*` | `ENABLED`, `API_KEY`, `PUBLISHABLE_KEY`, `WEBHOOK_SECRET`, success/cancel/portal URLs, price IDs |
| `MOBILE_MONEY__*` | MTN + Airtel keys/secrets, `MTN_ENVIRONMENT`, `UGX_PER_USD` |
| `FLUTTERWAVE__*` | `ENABLED`, `SECRET_KEY`, `PUBLIC_KEY`, `SECRET_HASH` |
| `PUBLIC_APP_URL` | Origin for email links + Stripe redirects |

> For managed Postgres (Crane Cloud / RDS / Neon), point `DATABASE__URL` at it,
> set `EMBEDDED_POSTGRES__ENABLED=false`, **percent-encode** reserved characters
> in the password, and mirror `POSTGRES_HOST/PORT/USER` so the container can
> `pg_isready` + run migrations. See [`docs/26-database-migrations.md`](26-database-migrations.md).

## Frontend (Next.js)

Frontend variables are prefixed `NEXT_PUBLIC_` and **baked into the browser
bundle at build time** (never put secrets here). Defined in `frontend/.env.local`
(dev) and `frontend/.env.production`.

| Variable | Example | Notes |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8080` (dev) / `https://www.optiscan.makstartup.com` (prod) | Backend base URL. Left empty in the Docker build — nginx proxies `/api`. |

## CI / deploy secrets (GitHub Actions)

Not app config, but required by the workflows (set as repo/org secrets):
`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HF_TOKEN`, `HF_USER`, `DATABASE_URL`,
`CRANE_CLOUD_EMAIL`/`_PASSWORD`/`_CPU_APP_ID`/`_GPU_APP_ID`/`_CPU_URL`. See
[`docs/25-ci-cd-pipeline.md`](25-ci-cd-pipeline.md).

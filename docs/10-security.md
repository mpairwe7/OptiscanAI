# Security

JWT authentication, rate limiting, request tracing, dependency scanning, container scanning, and SBOM generation.

## Authentication

The API uses JWT (JSON Web Token) authentication, toggleable via `AUTH_ENABLED` environment variable.

### Configuration

| Setting | Default | Description |
|---|---|---|
| `AUTH_ENABLED` | `false` | Enable/disable JWT authentication |
| `JWT_SECRET` | `change-me-in-production-use-env-var` | HMAC signing secret |
| `JWT_EXPIRY_SECONDS` | `3600` | Token lifetime (1 hour) |

### Getting a Token

```bash
# Exchange API key for JWT
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-secret-key"}'

# Response
{"access_token": "eyJ...", "token_type": "bearer", "expires_in": 3600}
```

### Using a Token

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Authorization: Bearer eyJ..." \
  -F "file=@fundus_image.jpg"
```

### Role-Based Access

Tokens include a `role` field. The `require_role` dependency enforces role checks:

- `user` — Standard prediction access
- `admin` — Full access including configuration

When `AUTH_ENABLED=false`, all requests are treated as admin (development mode).

### Implementation

- **Module**: `backend/app/core/auth.py`
- **Token format**: HMAC-SHA256 signed (header.payload.signature)
- **Router**: `backend/app/routers/auth.py` — `/api/v1/auth/token`

## Rate Limiting

In-memory per-IP rate limiter prevents API abuse.

| Setting | Default | Description |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |

Returns HTTP 429 when limit exceeded. Implementation: `backend/app/middleware/rate_limit.py`

## Request Tracing

Every request is assigned a unique `X-Request-ID` header for end-to-end tracing.

- Client can pass `X-Request-ID` header; otherwise one is auto-generated (UUID4)
- Response includes `X-Request-ID` and `X-Response-Time-Ms` headers
- All logs include request_id for correlation
- Prediction logs link to request_id for audit

Implementation: `backend/app/middleware/request_id.py`

## Structured Logging

Production logging outputs JSON lines for structured log aggregation.

| Setting | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` for structured, `text` for human-readable |

### JSON Log Format

```json
{
  "timestamp": "2026-04-25T10:30:00+00:00",
  "level": "INFO",
  "logger": "backend.app.middleware.request_id",
  "message": "POST /api/v1/predict 200 45.2ms",
  "module": "request_id",
  "request_id": "abc-123-def",
  "latency_ms": 45.2,
  "endpoint": "/api/v1/predict",
  "status_code": 200
}
```

Implementation: `backend/app/core/logging_config.py`

## Prediction Logging

Every prediction is logged to daily-rotated JSONL files for audit and drift analysis.

```
logs/predictions/predictions_2026-04-25.jsonl
```

Each entry contains:
- `request_id`, `user`, `timestamp`
- `threshold`, `inference_ms`, `model_loaded`
- `image_width`, `image_height`
- `num_detected`, `referral_priority`
- `top_predictions` (top 5 with codes and probabilities)

Implementation: `backend/app/core/prediction_logger.py`

## CI Security Scanning

The security scan workflow (`.github/workflows/security-scan.yml`) runs on every push/PR and weekly:

### Dependency Scanning
- **pip-audit**: Scans Python dependencies for known vulnerabilities
- **TruffleHog**: Detects committed secrets and API keys

### Container Scanning
- **Trivy**: Scans Docker images for CRITICAL and HIGH vulnerabilities
- Results uploaded as SARIF to GitHub Security tab

### SBOM Generation
- **Anchore**: Generates Software Bill of Materials (SPDX format)
- Required for EU AI Act compliance (medical AI transparency)
- Uploaded as CI artifact

### Schedule

| Trigger | When |
|---|---|
| Push | Every push to `main` or `develop` |
| PR | Every pull request to `main` |
| Schedule | Weekly (Monday 6am UTC) |

## Key Files

| File | Purpose |
|---|---|
| `backend/app/core/auth.py` | JWT token creation, validation, role enforcement |
| `backend/app/core/logging_config.py` | JSON structured logging setup |
| `backend/app/core/prediction_logger.py` | Append-only prediction audit log |
| `backend/app/middleware/rate_limit.py` | Per-IP rate limiting |
| `backend/app/middleware/request_id.py` | Request ID tracing + latency headers |
| `backend/app/routers/auth.py` | Token exchange endpoint |
| `.github/workflows/security-scan.yml` | Dependency, container, SBOM CI pipeline |

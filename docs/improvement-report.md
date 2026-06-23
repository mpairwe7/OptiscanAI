# Improvement Suggestions

## 1. Backend Architecture & Code

| Area | Current Observation | Suggested Improvement |
|------|---------------------|----------------------|
| **Entry‑point (`main.py`)** | Uses `lifespan` to initialise many services. Errors are caught and logged as *non‑fatal* but the app continues. | Convert critical failures (e.g., model loading, DB connection) to fatal errors that abort startup. This prevents the API from serving partially functional endpoints.
| **Router Design (`predict.py`)** | Contains a large monolithic endpoint with image validation, gating, inference, OOD checks, logging, event‑bus, billing, and response construction. | Split responsibilities into dedicated service classes (e.g., `ImageValidator`, `FundusGateService`, `InferenceService`). This makes the endpoint easier to test and audit.
| **Error Messages** | Directly returns raw exception strings (`Invalid image file: {e}`). | Wrap internal errors in generic client‑facing messages and log the details. Avoid leaking stack traces or internal library paths to callers.
| **Dependency Injection** | Direct imports inside the endpoint (`from src.data.fundus_gate_v2 import gate_image`). | Use dependency injection via FastAPI `Depends` so that alternative implementations (e.g., mocks for tests) can be swapped without changing the router.
| **Rate Limiting** | Middleware is present but the implementation is not visible. | Ensure the limiter uses a distributed store (Redis) for multi‑instance deployments and returns `429` with `Retry‑After` header.
| **Request ID Propagation** | Generates a UUID when `request` is `None`. | Enforce a request‑ID header (`X-Request-ID`) from the client and propagate it to all downstream services (logging, event‑bus, monitoring) to guarantee traceability.
| **Logging** | Uses `logging` with custom `prediction_logger`. | Adopt structured JSON logging (e.g., `python-json-logger`) and include correlation IDs, user IDs, and latency metrics for easy ingestion by ELK/Datadog.
| **Security Headers** | Headers are added in a custom middleware. | Consider using the `SecureHeaders` library to ensure all recommended headers are set consistently and to avoid duplication.
| **Authentication** | `get_current_user` is used, but token validation details are hidden. | Enforce token revocation list and short‑lived access tokens. Store JWT secret in a vault (e.g., HashiCorp Vault) rather than environment variables.

## 2. Docker Image & Runtime

| Observation | Recommendation |
|------------|----------------|
| Multi‑stage build is solid, but the final image runs many services via `supervisord`. | Replace `supervisord` with a minimal process manager like `s6` or run separate containers for DB, Nginx, and the API. This aligns with the principle of *one process per container*.
| The image copies model weight files directly (`COPY models/...`). | Store large model artefacts in a separate volume or object store (S3, GCS) and download at container start. This reduces image size and enables model updates without rebuilding.
| Non‑root user `optiscan` is created, but several `RUN` commands still execute as root (e.g., package installs). | Use `USER root` only for the installation layer and switch to `USER optiscan` before copying application code. Add `--no-install-recommends` to reduce attack surface.
| Secrets (e.g., DB password, API keys) are passed via environment variables. | Use Docker secrets or a secrets manager (AWS Secrets Manager, GCP Secret Manager) and mount them at runtime. Avoid exposing them in `docker inspect` output.
| Healthcheck uses a simple `curl` to `/health`. | Extend healthcheck to also verify that the model is loaded (`/health?check=model`). This ensures the container is only marked healthy when inference is possible.

## 3. CI/CD Pipeline (`.github/workflows`)

| File | Current State | Suggested Enhancement |
|------|---------------|----------------------|
| `docker-publish.yml` | Builds and pushes Docker image. | Add *SBOM* generation (Syft or CycloneDX) and *image signing* (cosign) before publishing.
| `security-scan.yml` | Runs a security scan (unspecified). | Ensure the scan includes secret detection (GitGuardian), container scanning (Trivy), and dependency vulnerability checks. Fail the pipeline on any *high* severity finding.
| `ml-pipeline.yml` | Executes model training / registration. | Cache model artefacts between runs to avoid re‑training when source code hasn’t changed. Also add a step to upload the model to a model registry with versioning.
| `quantization.yml` | Quantises the model. | Run quantisation on a separate, isolated runner with limited network access to prevent supply‑chain attacks.
| `billing-smoke.yml` | Smoke tests billing endpoints. | Include *contract tests* (e.g., Pact) for all public APIs to catch breaking changes early.
| General | No explicit linting or type‑checking step. | Add a stage that runs `ruff` (lint) and `mypy` (type checking) with `--strict` flags. Fail on any warning.

## 4. LLM / Civitai Integration (found in `integrations/` – pending review)

| Potential Issue | Recommendation |
|----------------|----------------|
| API key (`CIVITAI_API_KEY`) may be read from environment variables directly. | Store the key in a secret manager and retrieve it at runtime. Rotate the key regularly and audit usage.
| Prompt construction is not shown but could concatenate user input directly. | Apply *prompt sanitisation*: escape or filter potentially harmful user content, enforce a maximum token length, and implement a *system prompt* that restricts the model’s behaviour (e.g., no disallowed content).
| No rate‑limiting on LLM calls. | Add a client‑side throttling layer (e.g., token bucket) before invoking the external LLM service.
| Lack of observability on LLM latency and errors. | Emit metrics (`latency_ms`, `error_rate`) to Prometheus and set alerts for abnormal spikes.
| No fallback when the LLM service is unavailable. | Implement a circuit‑breaker pattern that returns a graceful degradation response (e.g., `service_unavailable`) and logs the incident.

## 5. General Recommendations

1. **Threat Modeling** – Run the `determine-threat-model` skill to map entry points (public API, admin endpoints, CI secrets) and document data flow.
2. **Security Implementation Plan** – Use the `create-security-implementation-plan` skill to produce a verification checklist before any new code is merged.
3. **Automated Testing** – Achieve >90% coverage for critical paths (image validation, fundus gate, model inference). Include *property‑based tests* for image processing.
4. **Dependency Scanning** – Before adding new packages, invoke `scan_dependencies` to verify safety and approved versions.
5. **Documentation** – Enrich OpenAPI schema with response examples and clearer error codes. Add a *security* section describing auth flow and required headers.

---

**Next steps**
- Review the pending `agents.py` and `review.py` routers to complete the LLM flow trace.
- Apply the above suggestions incrementally, starting with secret management and Docker hardening.
- Run the security scanner (`run-security-scanner`) after each change to ensure no new findings are introduced.

*All file references above are clickable links for quick navigation.*

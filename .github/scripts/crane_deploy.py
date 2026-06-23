#!/usr/bin/env python3
"""Deploy OptiscanAI images to Crane Cloud and assert the managed-DB env vars.

Everything is read from the environment (set by the CI workflow from GitHub
secrets) so no credential is ever written to argv or committed to the repo:

  CRANE_CLOUD_API   Crane Cloud API base (default https://api.cranecloud.io)
  CRANE_EMAIL       login email (lowercase)
  CRANE_PASSWORD    login password
  CRANE_CPU_APP_ID  CPU app id on RENU            (optional)
  CRANE_GPU_APP_ID  GPU app id                    (optional)
  CPU_TAG / GPU_TAG SHA-pinned image refs to roll out
  DATABASE_URL      full asyncpg DSN for the managed Postgres — the only secret
                    that carries the DB password

For each app it PATCHes {"image": <tag>, "env_vars": {...}}. Crane Cloud MERGES
env_vars on PATCH, so this adds the DB config without disturbing MODEL_PATH,
DEVICE, etc. The pg_isready params (POSTGRES_HOST/PORT/USER) are derived from
DATABASE_URL so the container's backend-start.sh can run `alembic upgrade head`
against the managed DB. DATABASE_URL itself is never printed.

Stdlib only (urllib + json) — no pip install needed on the runner.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("CRANE_CLOUD_API", "https://api.cranecloud.io").rstrip("/")

# A 5xx from Crane Cloud's deploy endpoint is frequently transient (control-
# plane hiccup), so retry a few times with linear backoff before failing CI.
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds; multiplied by the attempt number


def _request(method, path, payload, token=None):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode() or "{}"
        return resp.status, json.loads(body)


def _error_detail(exc):
    """Return Crane's HTTP error body, with the DB DSN/password redacted.

    The bare status line ("HTTP 500 Internal Server Error") is useless for
    debugging — the real cause lives in the response body. We redact the DSN
    and its password first, in case a server-side stack trace echoes the
    request payload back to us (these must never reach CI logs).
    """
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 — diagnostics must never raise
        return ""
    if not body:
        return ""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if dsn:
        body = body.replace(dsn, "***")
        pwd = urllib.parse.urlparse(dsn).password
        if pwd:
            body = body.replace(pwd, "***").replace(urllib.parse.unquote(pwd), "***")
    return body[:1500]


def db_env_vars():
    """Build the managed-DB env block from DATABASE_URL (empty if unset)."""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        return {}
    parsed = urllib.parse.urlparse(dsn)
    return {
        "DATABASE__ENABLED": "true",
        "DATABASE__URL": dsn,
        # Force the external DB; silences the in-image embedded Postgres.
        "EMBEDDED_POSTGRES__ENABLED": "false",
        # backend-start.sh runs pg_isready against these before `alembic
        # upgrade head`; alembic itself reads DATABASE__URL.
        "POSTGRES_HOST": parsed.hostname or "",
        "POSTGRES_PORT": str(parsed.port or ""),
        "POSTGRES_USER": urllib.parse.unquote(parsed.username or ""),
    }


def main():
    email = os.environ["CRANE_EMAIL"]
    password = os.environ["CRANE_PASSWORD"]

    _, login = _request("POST", "/users/login", {"email": email, "password": password})
    token = login["data"]["access_token"]
    print("Logged in to Crane Cloud")

    env_vars = db_env_vars()
    if env_vars:
        shown = ", ".join(k for k in env_vars if k != "DATABASE__URL")
        print(f"Asserting managed-DB env: {shown} + DATABASE__URL(***)")
    else:
        print("::warning::DATABASE_URL not set — deploying image only, DB env unchanged")

    targets = [
        ("CPU", os.environ.get("CRANE_CPU_APP_ID", ""), os.environ.get("CPU_TAG", "")),
        ("GPU", os.environ.get("CRANE_GPU_APP_ID", ""), os.environ.get("GPU_TAG", "")),
    ]

    rc = 0
    for label, app_id, tag in targets:
        if not app_id:
            print(f"{label}: no app id configured, skipping")
            continue
        # Image is optional: on a deploy-only run the workflow leaves the tag
        # empty, so we PATCH env_vars alone (switch DB without a rebuild). On a
        # normal run the freshly built SHA-pinned tag rolls the pod over.
        payload = {}
        if tag:
            payload["image"] = tag
        if env_vars:
            payload["env_vars"] = env_vars
        if not payload:
            print(f"{label}: no image tag and no DB env — nothing to update, skipping")
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                status, _ = _request("PATCH", f"/apps/{app_id}", payload, token)
                print(f"{label} deploy ({tag or 'env-only'}): HTTP {status}")
                break
            except urllib.error.HTTPError as exc:
                detail = _error_detail(exc)
                suffix = f" — {detail}" if detail else ""
                # 5xx is usually a transient control-plane error: back off and
                # retry. 4xx is a contract problem — fail fast, no retry.
                if exc.code >= 500 and attempt < MAX_RETRIES:
                    print(
                        f"::warning::{label} deploy attempt {attempt}/{MAX_RETRIES}: "
                        f"HTTP {exc.code} {exc.reason}{suffix}; retrying"
                    )
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                print(f"::error::{label} deploy failed: HTTP {exc.code} {exc.reason}{suffix}")
                rc = 1
                break
            except Exception as exc:  # noqa: BLE001 — surface any failure to CI
                print(f"::error::{label} deploy failed: {exc}")
                rc = 1
                break
    return rc


if __name__ == "__main__":
    sys.exit(main())

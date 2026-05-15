"""End-to-end smoke for the Phase 6 subscription billing platform.

Codifies the walkthrough we ran by hand in docs/23-billing-platform.md § 11:
sign-up → quota exhaustion → paywall payload → feature gate → seat limit →
seat purchase simulation → Free downgrade guard → admin webhook list.

Requirements:
  - A Postgres reachable at $TEST_DATABASE_URL (defaults to the dev sidecar at
    127.0.0.1:55433). The test creates a clean schema by running `alembic
    upgrade head`, then truncates between tests.
  - The optional billing deps from pyproject.toml (sqlalchemy[asyncio],
    asyncpg, alembic, argon2-cffi, pyjwt, stripe, httpx, email-validator).
    If any of these is missing the module is skipped wholesale.

The whole module skips gracefully when the DB isn't reachable, so it's safe
to keep in CI behind an opt-in marker.

Run locally:
    DATABASE__ENABLED=true \\
    DATABASE__URL=postgresql+asyncpg://optiscan:optiscan@127.0.0.1:55433/optiscan \\
    BILLING__ENABLED=true AUTH_ENABLED=true \\
    JWT_SECRET=test-secret-32+-chars \\
    uv run pytest tests/test_billing_smoke.py -v
"""
from __future__ import annotations

import os
import uuid as _uuid
from datetime import datetime, timedelta, timezone

import pytest

# ── Optional-dep skips ────────────────────────────────────────────────────────

pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")
pytest.importorskip("alembic")
pytest.importorskip("argon2", reason="needs argon2-cffi for password hashing")
pytest.importorskip("jwt", reason="needs pyjwt")
pytest.importorskip("httpx")

import httpx  # noqa: E402

# ── Settings must be set BEFORE the app is imported ───────────────────────────

_DEFAULT_DB = "postgresql+asyncpg://optiscan:optiscan@127.0.0.1:55433/optiscan"
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", _DEFAULT_DB)

os.environ.setdefault("DATABASE__ENABLED", "true")
os.environ.setdefault("DATABASE__URL", TEST_DB_URL)
os.environ.setdefault("BILLING__ENABLED", "true")
os.environ.setdefault("AUTH_ENABLED", "true")
os.environ.setdefault("JWT_SECRET", "billing-smoke-test-secret-9b14c3a87f5e6d29")
os.environ.setdefault("PUBLIC_APP_URL", "http://test.localhost")
os.environ.setdefault("EMAIL__ENABLED", "false")
os.environ.setdefault("EMAIL__PROVIDER", "console")

# Health-check the test DB before importing the app — gives a clean skip
# instead of a 500-ms uvicorn boot failure. Sync socket check avoids
# leaking an event loop across pytest-asyncio's loop scope.
import re
import socket


def _port_open(url: str, timeout: float = 3.0) -> bool:
    m = re.match(r".+@([^:/]+):(\d+)/", url)
    if not m:
        return False
    host, port = m.group(1), int(m.group(2))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if not _port_open(TEST_DB_URL):
    pytest.skip(
        f"Test Postgres not reachable at {TEST_DB_URL}. "
        f"Start the dev sidecar: docker run -d --name optiscan-pg-test "
        f"-e POSTGRES_USER=optiscan -e POSTGRES_PASSWORD=optiscan "
        f"-e POSTGRES_DB=optiscan -p 127.0.0.1:55433:5432 postgres:16-alpine",
        allow_module_level=True,
    )

# Now safe to import — Settings will read the env vars above
from backend.app.core.config import settings  # noqa: E402
from backend.app.core.db import dispose_engine, init_engine, session_factory  # noqa: E402
from backend.app.main import app  # noqa: E402

import asyncio  # noqa: E402

pytestmark = pytest.mark.asyncio


# ── Schema setup / cleanup ────────────────────────────────────────────────────


def _truncate_sql() -> str:
    """Reset every billing-related table; keep the plan seed."""
    return """
    TRUNCATE TABLE
        usage_events,
        webhook_events,
        renewal_reminders,
        payment_intents,
        invoices,
        organization_invites,
        memberships,
        refresh_tokens,
        email_verification_tokens,
        password_reset_tokens,
        magic_link_tokens,
        subscriptions,
        organizations,
        users
    RESTART IDENTITY CASCADE;
    """


async def _ensure_schema() -> None:
    """Run alembic upgrade head on the test DB (idempotent)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "backend", "alembic.ini"))
    cfg.set_main_option("script_location", "backend/alembic")
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    # Alembic's runner is synchronous — run it on a thread to avoid blocking
    # the active event loop.
    await asyncio.to_thread(command.upgrade, cfg, "head")


@pytest.fixture(scope="session", autouse=True)
async def _engine_lifecycle():
    """Apply schema, init engine, dispose on session exit."""
    init_engine()
    await _ensure_schema()
    yield
    await dispose_engine()


@pytest.fixture(autouse=True)
async def _truncate_between_tests():
    """Clean state before every test."""
    factory = session_factory()
    assert factory is not None, "DB engine not initialized"
    async with factory() as db:
        await db.execute(_text(_truncate_sql()))
        await db.commit()
    yield


def _text(sql: str):
    """Late-imported sa.text — avoids hitting the import at module load."""
    from sqlalchemy import text
    return text(sql)


# ── HTTP client + helpers ─────────────────────────────────────────────────────


@pytest.fixture
async def client():
    """ASGI client bound directly to the app — bypasses uvicorn + lifespan,
    which means the heavy ML model is never loaded."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
def tiny_jpeg() -> bytes:
    """Real 32×32 JPEG bytes — passes PIL's structural integrity check that
    runs before the quota/feature-gate logic in /predict and /explain/*."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), color=(120, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


async def _sql(query: str, **params):
    factory = session_factory()
    async with factory() as db:
        result = await db.execute(_text(query), params)
        await db.commit()
        try:
            return result.fetchall()
        except Exception:
            return None


async def _exec(query: str, **params):
    """Execute without trying to fetch — for DML."""
    factory = session_factory()
    async with factory() as db:
        await db.execute(_text(query), params)
        await db.commit()


async def register(client: httpx.AsyncClient, email: str, *, password: str = "TestPass2026!", full_name: str = "Test User") -> dict:
    r = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "full_name": full_name},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def promote_to_practice(email: str) -> None:
    """Direct DB write — simulates what apply_stripe_subscription_event does
    when a checkout.session.completed webhook fires."""
    await _exec("""
        UPDATE subscriptions s
        SET plan_id=(SELECT id FROM plans WHERE code='practice'),
            provider='stripe',
            stripe_subscription_id='sub_test_practice',
            stripe_customer_id='cus_test',
            billing_cycle='monthly',
            current_period_start=now(),
            current_period_end=now() + interval '30 days'
        FROM organizations o, users u
        WHERE s.organization_id=o.id AND o.owner_user_id=u.id
        AND u.email_normalized=:email
    """, email=email.lower().strip())


async def promote_to_superuser(email: str) -> None:
    await _exec(
        "UPDATE users SET is_superuser=true, email_verified_at=now() WHERE email_normalized=:e",
        e=email.lower().strip(),
    )


async def force_exhaust_quota(email: str, count: int) -> None:
    """Insert `count` scan events to push the org against its quota."""
    await _exec("""
        INSERT INTO usage_events (organization_id, user_id, event_type, quantity, occurred_at)
        SELECT s.organization_id, u.id, 'scan', 1, now()
        FROM subscriptions s
        JOIN organizations o ON s.organization_id=o.id
        JOIN users u ON o.owner_user_id=u.id
        CROSS JOIN generate_series(1, :n)
        WHERE u.email_normalized=:e
    """, n=count, e=email.lower().strip())


async def org_id_for(email: str) -> str:
    rows = await _sql("""
        SELECT o.id FROM organizations o
        JOIN users u ON o.owner_user_id=u.id
        WHERE u.email_normalized=:e
    """, e=email.lower().strip())
    return str(rows[0][0])


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_register_creates_user_org_and_free_subscription(client):
    body = await register(client, "alice@example.com")
    user = body["user"]
    assert user["email"] == "alice@example.com"
    assert user["organization"]["is_personal"] is True
    assert user["subscription"]["plan"]["code"] == "free"
    assert user["subscription"]["plan"]["scan_limit_monthly"] == 10
    assert user["role"] == "owner"
    # Cookies set so subsequent calls authenticate
    assert "os_access" in client.cookies
    assert "os_refresh" in client.cookies


async def test_login_round_trips_after_register(client):
    await register(client, "bob@example.com", password="BobPass2026!")
    client.cookies.clear()
    r = await client.post("/api/v1/auth/login", json={
        "email": "bob@example.com", "password": "BobPass2026!",
    })
    assert r.status_code == 200, r.text
    me = (await client.get("/api/v1/auth/me")).json()
    assert me["email"] == "bob@example.com"


async def test_plans_catalog_lists_four_tiers(client):
    r = await client.get("/api/v1/billing/plans")
    assert r.status_code == 200
    codes = {p["code"] for p in r.json()}
    assert codes == {"free", "clinician", "practice", "health_system"}


async def test_quota_exceeded_returns_402_with_paywall_payload(client, tiny_jpeg):
    """The exact shape the frontend PaywallModal consumes."""
    await register(client, "cara@example.com")
    await promote_to_practice("cara@example.com")
    await force_exhaust_quota("cara@example.com", 3000)

    r = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", tiny_jpeg, "image/jpeg")},
    )
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "quota_exceeded"
    assert detail["plan"]["code"] == "practice"
    assert detail["plan"]["scan_limit_monthly"] == 3000
    assert detail["usage"]["used"] == 3000
    assert detail["usage"]["limit"] == 3000
    assert "resets_at" in detail["usage"]
    assert detail["upgrade_url"] == "/pricing"
    assert detail["recommended_plan"] == "health_system"
    # Response headers mirror the body
    assert r.headers.get("X-Usage-Used") == "3000"
    assert r.headers.get("X-Usage-Limit") == "3000"


async def test_feature_gate_blocks_shap_on_free(client, tiny_jpeg):
    """The exact 403 payload the frontend UpsellSheet consumes."""
    await register(client, "dan@example.com")
    r = await client.post(
        "/api/v1/explain/shap",
        files={"file": ("scan.jpg", tiny_jpeg, "image/jpeg")},
    )
    assert r.status_code == 403, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "feature_locked"
    assert detail["required_plan"] == "clinician"
    assert detail["current_plan"] == "free"
    assert detail["upgrade_url"] == "/pricing"


async def test_invite_seat_limit_guard(client):
    """Practice seats=5 included. After 4 pending invites, the 5th hits
    the 402 seat_limit_reached payload (used+pending=1+4 already equals 5)."""
    await register(client, "eve@example.com", full_name="Eve Practice")
    await promote_to_practice("eve@example.com")
    org_id = await org_id_for("eve@example.com")

    for i in range(4):
        r = await client.post(
            f"/api/v1/orgs/{org_id}/invites",
            json={"email": f"member{i}@hospital.org", "role": "clinician"},
        )
        assert r.status_code == 201, f"invite {i}: {r.text}"

    # 5th invite — owner(1) + 4 pending = 5 hits the limit
    r = await client.post(
        f"/api/v1/orgs/{org_id}/invites",
        json={"email": "overflow@hospital.org", "role": "clinician"},
    )
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "seat_limit_reached"
    assert detail["plan"]["effective_seat_limit"] == 5
    assert detail["plan"]["additional_seats"] == 0
    assert detail["upgrade_url"] == "/app/team"


async def test_simulated_seat_purchase_lifts_limit(client):
    """Webhook-set additional_seats raises the effective limit, unblocking
    further invites without code redeploy."""
    await register(client, "fran@example.com")
    await promote_to_practice("fran@example.com")
    org_id = await org_id_for("fran@example.com")

    # Fill to the included-seat limit
    for i in range(4):
        await client.post(
            f"/api/v1/orgs/{org_id}/invites",
            json={"email": f"mem{i}@hospital.org", "role": "clinician"},
        )

    # Simulate a Stripe customer.subscription.updated event adding 3 seats
    await _exec("""
        UPDATE subscriptions SET additional_seats=3, stripe_seat_item_id='si_test'
        WHERE organization_id=:o
    """, o=org_id)

    seats = (await client.get("/api/v1/billing/seats")).json()
    assert seats["additional_seats"] == 3
    assert seats["effective_limit"] == 8

    # Previously-403'ing 5th invite now succeeds
    r = await client.post(
        f"/api/v1/orgs/{org_id}/invites",
        json={"email": "former-overflow@hospital.org", "role": "clinician"},
    )
    assert r.status_code == 201, r.text


async def test_free_downgrade_blocked_when_members_exceed_target_seats(client):
    """Member-count guard: dropping a Practice org with 4 active members
    to Free (seat_limit=1) should 400 with a precise payload."""
    await register(client, "gina@example.com")
    await promote_to_practice("gina@example.com")
    org_id = await org_id_for("gina@example.com")

    # Create 3 real users + add them as active members so seat_count = 4
    # (owner + 3 extras). FK to users.id is satisfied because we INSERT
    # the user rows first in the same CTE.
    await _exec("""
        WITH new_users AS (
            INSERT INTO users (id, email, email_normalized, password_hash, full_name,
                               is_active, is_superuser, created_at, updated_at)
            SELECT gen_random_uuid(),
                   'gina-extra' || g || '@example.com',
                   'gina-extra' || g || '@example.com',
                   'argon2-stub',
                   'Extra ' || g,
                   true, false, now(), now()
            FROM generate_series(1, 3) g
            RETURNING id
        )
        INSERT INTO memberships (id, user_id, organization_id, role, status, accepted_at, created_at)
        SELECT gen_random_uuid(), id, :o, 'clinician', 'active', now(), now()
        FROM new_users
    """, o=org_id)

    r = await client.post(
        "/api/v1/billing/subscription/change",
        json={"plan_code": "free", "billing_cycle": "monthly"},
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "members_exceed_target_seats"
    assert detail["active_members"] == 4
    assert detail["target_seat_limit"] == 1


async def test_admin_endpoints_require_superuser(client):
    """Non-superuser hitting /admin/* gets 403 with a clear payload."""
    await register(client, "hank@example.com")
    r = await client.get("/api/v1/billing/admin/webhook-events")
    assert r.status_code == 403
    assert "Superuser" in r.json()["detail"]


async def test_admin_lists_webhook_events_after_promotion(client):
    await register(client, "ivy@example.com")
    await promote_to_superuser("ivy@example.com")
    # Cookie still valid; the auth context re-reads is_superuser from the DB
    r = await client.get("/api/v1/billing/admin/webhook-events")
    assert r.status_code == 200, r.text
    assert r.json() == []


async def test_admin_run_renewal_reminders_returns_summary(client):
    await register(client, "jay@example.com")
    await promote_to_superuser("jay@example.com")
    r = await client.post("/api/v1/billing/admin/run-renewal-reminders")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "ok"
    assert "reminders_sent" in payload
    assert "found_subscriptions" in payload


async def test_renewal_cron_picks_up_momo_sub_at_3_days_to_expiry(client):
    """A MoMo subscription with current_period_end inside the 3-day window
    should produce exactly one '3d' reminder row on first run, zero on second."""
    await register(client, "kim@example.com", full_name="Dr Kim Test")
    await promote_to_superuser("kim@example.com")
    await _exec("""
        UPDATE subscriptions SET
            plan_id=(SELECT id FROM plans WHERE code='clinician'),
            provider='mtn',
            billing_cycle='monthly',
            current_period_start=now() - interval '27 days',
            current_period_end=now() + interval '3 days' - interval '1 hour'
        FROM organizations o, users u
        WHERE subscriptions.organization_id=o.id AND o.owner_user_id=u.id
        AND u.email_normalized='kim@example.com'
    """)

    # First run sends one reminder
    r1 = (await client.post("/api/v1/billing/admin/run-renewal-reminders")).json()
    assert r1["reminders_sent"] == 1, r1
    assert r1["found_subscriptions"] == 1

    rows = await _sql(
        "SELECT kind FROM renewal_reminders WHERE error IS NULL ORDER BY sent_at DESC"
    )
    assert len(rows) == 1
    assert str(rows[0][0]).lower() == "3d"

    # Idempotency: second run skips
    r2 = (await client.post("/api/v1/billing/admin/run-renewal-reminders")).json()
    assert r2["reminders_sent"] == 0, r2
    assert r2["reminders_skipped"] == 1


async def test_usage_endpoint_breakdown_includes_scan_count(client):
    await register(client, "lee@example.com")
    await promote_to_practice("lee@example.com")
    await force_exhaust_quota("lee@example.com", 7)
    usage = (await client.get("/api/v1/billing/usage")).json()
    assert usage["scans_used"] == 7
    assert usage["scan_limit"] == 3000
    assert usage["scans_remaining"] == 2993
    assert usage["breakdown"]["scan"] == 7

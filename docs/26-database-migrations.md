# Database & Migrations

The SaaS/billing layer (Phase 6) uses **PostgreSQL** via async SQLAlchemy
(`asyncpg`) with **Alembic** migrations. The DB is only needed when
`DATABASE__ENABLED=true` (default for the Compose/K8s stacks).

## Layout

```
backend/
├── alembic.ini              # script_location=backend/alembic, prepend_sys_path=.
└── alembic/
    ├── env.py               # async engine; overrides URL from settings.database.url
    ├── script.py.mako
    └── versions/
        ├── 0001_initial_billing_schema.py
        ├── 0002_renewal_reminders.py
        └── 0003_additional_seats.py
```

The connection URL is **sourced from app settings** (`settings.database.url`,
i.e. env `DATABASE__URL`) — `env.py` overrides `sqlalchemy.url` so there is a
single source of truth. The literal URL in `alembic.ini` is only a local
fallback.

## Commands

Run from the **repo root** and pass the ini explicitly (its `script_location` is
resolved from the current directory):

```bash
# Apply all pending migrations
uv run alembic -c backend/alembic.ini upgrade head

# Autogenerate a migration after editing models
uv run alembic -c backend/alembic.ini revision --autogenerate -m "describe change"

# Roll back one revision
uv run alembic -c backend/alembic.ini downgrade -1

# Inspect
uv run alembic -c backend/alembic.ini current
uv run alembic -c backend/alembic.ini history
```

Set `DATABASE__URL` first, e.g.:

```bash
export DATABASE__URL="postgresql+asyncpg://optiscan:optiscan@localhost:5432/optiscan"
```

## In containers

`scripts/container/backend-start.sh` waits for Postgres (`pg_isready`) and then
runs `alembic -c backend/alembic.ini upgrade head` before exec'ing uvicorn —
migrations are applied automatically on startup and are idempotent. If the
upgrade fails the backend still starts, but billing endpoints will return 5xx.

## Managed Postgres (Crane Cloud / RDS / Neon)

- Use the `postgresql+asyncpg://` driver.
- **Percent-encode** reserved characters in the password (`@`→`%40`, `#`→`%23`,
  `>`→`%3E`, `%`→`%25`, …) — SQLAlchemy decodes the URL userinfo.
- Mirror `POSTGRES_HOST/PORT/USER` so the container's `pg_isready` check matches
  the URL, and set `EMBEDDED_POSTGRES__ENABLED=false`.
- Append `?ssl=require` to enforce TLS.

See [`docs/22-crane-cloud-deployment.md`](22-crane-cloud-deployment.md) for the
three Postgres shapes (sidecar / embedded / managed) and
[`docs/24-environment-variables.md`](24-environment-variables.md) for the full
`DATABASE__*` surface.

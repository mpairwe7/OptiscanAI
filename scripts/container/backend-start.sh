#!/usr/bin/env bash
# Backend startup wrapper.
#
#   1. If DATABASE__ENABLED=true, wait for the in-container Postgres to be
#      ready and run `alembic upgrade head` (idempotent — applies only what's
#      missing).
#   2. Exec uvicorn so supervisord supervises the actual server, not this
#      wrapper.

set -euo pipefail

log() { printf '[backend-start] %s\n' "$*" >&2; }

API_PORT="${API_PORT:-8081}"
PG_HOST="${POSTGRES_HOST:-127.0.0.1}"
PG_PORT="${POSTGRES_PORT:-5432}"
PG_USER="${POSTGRES_USER:-optiscan}"
WAIT_MAX="${POSTGRES_WAIT_MAX:-60}"   # seconds

if [ "${DATABASE__ENABLED:-false}" = "true" ]; then
  log "DATABASE__ENABLED=true — waiting up to ${WAIT_MAX}s for Postgres at ${PG_HOST}:${PG_PORT}"
  i=0
  until /usr/bin/pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -q; do
    i=$((i + 1))
    if [ "$i" -ge "$WAIT_MAX" ]; then
      log "Postgres did not become ready in ${WAIT_MAX}s — starting backend anyway (will 503 on billing endpoints)"
      break
    fi
    sleep 1
  done

  if /usr/bin/pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -q; then
    log "Postgres ready after ${i}s — running alembic migrations"
    cd /app
    if ! /opt/venv/bin/alembic upgrade head; then
      log "alembic upgrade failed — backend will start, but billing endpoints will 5xx"
    fi
  fi
fi

log "Starting uvicorn on port ${API_PORT}"
exec /opt/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port "${API_PORT}"

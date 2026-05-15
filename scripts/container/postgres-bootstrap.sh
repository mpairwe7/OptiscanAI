#!/usr/bin/env bash
# Initialize the embedded Postgres cluster on first boot, then exec the
# postgres process so supervisord can supervise it.
#
# Idempotent — re-running on an existing cluster just starts it.
#
# Used by the in-container Postgres (Option 1, development / single-tenant
# pilots). For production multi-tenant deployments, move to a sidecar (Option
# 2) or managed Postgres (Option 3) — see docs/23-billing-platform.md § 14.

set -euo pipefail

# Allow the embedded cluster to be silenced when a sidecar/managed Postgres
# is in use (Options 2 + 3). supervisord still keeps the program alive via
# `sleep infinity` so the [program:postgres] block stays "running" with zero
# CPU; this avoids supervisord retry storms.
if [ "${EMBEDDED_POSTGRES__ENABLED:-true}" = "false" ]; then
  printf '[postgres-bootstrap] EMBEDDED_POSTGRES__ENABLED=false — sleeping; expect external Postgres at %s\n' \
    "${POSTGRES_HOST:-unknown}:${POSTGRES_PORT:-5432}" >&2
  exec sleep infinity
fi

PG_BIN="${PG_BIN:-/usr/lib/postgresql/14/bin}"
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PG_USER="${POSTGRES_USER:-optiscan}"
PG_PASSWORD="${POSTGRES_PASSWORD:-optiscan}"
PG_DATABASE="${POSTGRES_DB:-optiscan}"
PG_LISTEN="${POSTGRES_LISTEN:-127.0.0.1}"
PG_PORT="${POSTGRES_PORT:-5432}"

log() { printf '[postgres-bootstrap] %s\n' "$*" >&2; }

# Ensure the data dir exists and is owned by postgres
mkdir -p "$PGDATA"
chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  log "Initializing new cluster at $PGDATA"
  su -s /bin/bash postgres -c "$PG_BIN/initdb \
    --pgdata=$PGDATA \
    --encoding=UTF8 \
    --locale=C.UTF-8 \
    --auth-host=scram-sha-256 \
    --auth-local=trust \
    --username=postgres"

  # Loopback-only; the app talks to us at 127.0.0.1:5432 inside the container.
  printf "listen_addresses = '%s'\nport = %s\nlog_destination = 'stderr'\nlog_statement = 'none'\nshared_buffers = 128MB\nmax_connections = 50\n" \
    "$PG_LISTEN" "$PG_PORT" >> "$PGDATA/postgresql.conf"

  # Local md5/scram for the optiscan user
  printf "host all %s 127.0.0.1/32 scram-sha-256\nhost all %s ::1/128 scram-sha-256\n" \
    "$PG_USER" "$PG_USER" >> "$PGDATA/pg_hba.conf"

  log "Starting temporary cluster to provision role + database"
  su -s /bin/bash postgres -c "$PG_BIN/pg_ctl -D $PGDATA -l /tmp/pg-init.log -o '-c listen_addresses=127.0.0.1 -p $PG_PORT' -w start"

  su -s /bin/bash postgres -c "$PG_BIN/psql -p $PG_PORT -v ON_ERROR_STOP=1 <<SQL
CREATE USER $PG_USER WITH PASSWORD '$PG_PASSWORD';
CREATE DATABASE $PG_DATABASE OWNER $PG_USER;
ALTER USER $PG_USER WITH SUPERUSER;
GRANT ALL PRIVILEGES ON DATABASE $PG_DATABASE TO $PG_USER;
SQL"

  log "Stopping temporary cluster"
  su -s /bin/bash postgres -c "$PG_BIN/pg_ctl -D $PGDATA -m fast -w stop"
  log "Cluster bootstrap complete"
else
  log "Existing cluster at $PGDATA (version $(cat "$PGDATA/PG_VERSION")) — skipping init"
fi

# Hand off to supervisord — exec so signals propagate cleanly
log "Starting postgres in foreground"
exec su -s /bin/bash postgres -c "$PG_BIN/postgres -D $PGDATA -c config_file=$PGDATA/postgresql.conf"

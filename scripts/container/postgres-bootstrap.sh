#!/usr/bin/env bash
# Initialize + run the embedded Postgres cluster as the NON-ROOT app user.
#
# Hardened PaaS platforms (e.g. Crane Cloud) run the container as an
# unprivileged user, so supervisord cannot `su` to the system `postgres`
# account — the old root-only bootstrap silently failed there and the cluster
# never came up. PostgreSQL itself refuses to run as root anyway, so we always
# run as the app user (default `optiscan`), which must own PGDATA (the Dockerfile
# chowns /var/lib/postgresql to it). When invoked directly as root (plain local
# `docker run`), we drop to the app user first.
#
# Idempotent — re-running on an existing cluster just starts it.
#
# Single-tenant / dev / pilot only. For multi-tenant production, point
# DATABASE__URL at a managed Postgres and set EMBEDDED_POSTGRES__ENABLED=false.

set -euo pipefail

log() { printf '[postgres-bootstrap] %s\n' "$*" >&2; }

# Silence the embedded cluster when an external/managed Postgres is in use.
if [ "${EMBEDDED_POSTGRES__ENABLED:-true}" = "false" ]; then
  log "EMBEDDED_POSTGRES__ENABLED=false — sleeping; expect external Postgres at ${POSTGRES_HOST:-unknown}:${POSTGRES_PORT:-5432}"
  exec sleep infinity
fi

APP_USER="${POSTGRES_RUN_AS:-optiscan}"

# Postgres won't run as root. If we were started as root (local docker run),
# re-exec as the unprivileged app user so everything below runs unprivileged.
if [ "$(id -u)" = "0" ]; then
  log "running as root — dropping to '$APP_USER'"
  exec su -s /bin/bash "$APP_USER" -c "$(printf '%q ' "$0" "$@")"
fi

PG_BIN="${PG_BIN:-/usr/lib/postgresql/15/bin}"
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PG_USER="${POSTGRES_USER:-optiscan}"
PG_PASSWORD="${POSTGRES_PASSWORD:-optiscan}"
PG_DATABASE="${POSTGRES_DB:-optiscan}"
PG_LISTEN="${POSTGRES_LISTEN:-127.0.0.1}"
PG_PORT="${POSTGRES_PORT:-5432}"
# /var/run/postgresql is root-owned; use a dir the app user can write.
PG_SOCK="${POSTGRES_SOCKET_DIR:-/tmp}"

export PATH="$PG_BIN:$PATH"

mkdir -p "$PGDATA"
chmod 700 "$PGDATA" 2>/dev/null || true

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  log "Initializing new cluster at $PGDATA (uid=$(id -u), superuser role=$PG_USER)"
  # The OS user we run as becomes the bootstrap superuser role ($PG_USER), so
  # local-socket (trust) connections authenticate without a password.
  "$PG_BIN/initdb" \
    --pgdata="$PGDATA" \
    --encoding=UTF8 \
    --locale=C.UTF-8 \
    --auth-host=scram-sha-256 \
    --auth-local=trust \
    --username="$PG_USER"

  {
    printf "listen_addresses = '%s'\n" "$PG_LISTEN"
    printf "port = %s\n" "$PG_PORT"
    printf "unix_socket_directories = '%s'\n" "$PG_SOCK"
    printf "log_destination = 'stderr'\n"
    printf "log_statement = 'none'\n"
    printf "shared_buffers = 128MB\n"
    printf "max_connections = 50\n"
  } >> "$PGDATA/postgresql.conf"

  # App connects over TCP from 127.0.0.1 with a password (scram).
  printf "host all %s 127.0.0.1/32 scram-sha-256\nhost all %s ::1/128 scram-sha-256\n" \
    "$PG_USER" "$PG_USER" >> "$PGDATA/pg_hba.conf"

  log "Starting temporary cluster to set password + create database"
  "$PG_BIN/pg_ctl" -D "$PGDATA" -l /tmp/pg-init.log \
    -o "-c unix_socket_directories=$PG_SOCK -c listen_addresses=127.0.0.1 -p $PG_PORT" -w start

  "$PG_BIN/psql" -h "$PG_SOCK" -p "$PG_PORT" -U "$PG_USER" -d postgres -v ON_ERROR_STOP=1 <<SQL
ALTER USER $PG_USER WITH PASSWORD '$PG_PASSWORD';
CREATE DATABASE $PG_DATABASE OWNER $PG_USER;
SQL

  "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop
  log "Cluster bootstrap complete"
else
  log "Existing cluster at $PGDATA (version $(cat "$PGDATA/PG_VERSION")) — skipping init"
fi

log "Starting postgres in foreground (uid=$(id -u))"
exec "$PG_BIN/postgres" -D "$PGDATA" -c config_file="$PGDATA/postgresql.conf"

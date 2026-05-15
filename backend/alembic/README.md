# Alembic migrations

```bash
# Apply latest
uv run alembic upgrade head

# Create new migration (after editing models)
uv run alembic revision --autogenerate -m "describe change"

# Rollback one
uv run alembic downgrade -1
```

URL is sourced from `settings.database.url` (env `DATABASE__URL`).

# Alembic migrations

Run from the repo root and pass the ini explicitly (`script_location` is resolved
from the current directory):

```bash
# Apply latest
uv run alembic -c backend/alembic.ini upgrade head

# Create new migration (after editing models)
uv run alembic -c backend/alembic.ini revision --autogenerate -m "describe change"

# Rollback one
uv run alembic -c backend/alembic.ini downgrade -1
```

URL is sourced from `settings.database.url` (env `DATABASE__URL`). Full guide:
[`docs/26-database-migrations.md`](../../docs/26-database-migrations.md).

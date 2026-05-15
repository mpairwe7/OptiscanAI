"""Renewal-reminder cron entrypoint.

Invoke from system cron (or any scheduler) at most once per hour. Common
schedule: daily at 02:00 UTC.

  # /etc/cron.d/optiscan-renewals
  0 2 * * * uvuser cd /srv/optiscan && uv run python -m backend.app.cli.renewal_reminders

Exit code 0 on success (including "no candidates"). Non-zero on infrastructure
errors so cron logs surface them.

The runner is idempotent — re-running the same day is a no-op because each
reminder row has a unique constraint on ``(subscription_id, period_end, kind)``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from backend.app.core.config import settings
from backend.app.core.db import dispose_engine, init_engine, session_factory
from backend.app.core.logging_config import setup_logging
from backend.app.services.renewal_service import run_renewal_reminders

logger = logging.getLogger(__name__)


async def _main() -> int:
    setup_logging(settings.log_level, settings.log_format)
    if not settings.database.enabled:
        logger.error("Database disabled — set DATABASE__ENABLED=true to run reminders")
        return 1
    if not settings.billing.enabled:
        logger.warning("Billing disabled — renewal cron is a no-op")
        return 0

    init_engine()
    factory = session_factory()
    if factory is None:
        logger.error("Session factory not initialized")
        return 2

    try:
        async with factory() as db:
            result = await run_renewal_reminders(db)
    finally:
        await dispose_engine()

    summary = result.as_dict()
    # Single-line JSON so the cron mail body / logs are grep-friendly.
    print(json.dumps({"event": "renewal_reminders", **summary}))
    logger.info("Renewal cron: %s", summary)
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

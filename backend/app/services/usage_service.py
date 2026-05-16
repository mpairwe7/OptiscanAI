"""Usage metering and quota lookups.

The hot path is :func:`count_scans_in_period`. It uses the composite index
``(organization_id, event_type, occurred_at)`` defined on ``usage_events``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.subscription import Subscription
from backend.app.models.usage_event import UsageEvent, UsageEventType


async def count_events_in_period(
    db: AsyncSession,
    *,
    organization_id: str,
    event_type: UsageEventType,
    period_start: datetime,
    period_end: datetime,
) -> int:
    stmt = (
        select(func.count())
        .select_from(UsageEvent)
        .where(
            UsageEvent.organization_id == organization_id,
            UsageEvent.event_type == event_type,
            UsageEvent.occurred_at >= period_start,
            UsageEvent.occurred_at < period_end,
        )
    )
    return int((await db.execute(stmt)).scalar_one())


async def breakdown_by_event_type(
    db: AsyncSession,
    *,
    organization_id: str,
    period_start: datetime,
    period_end: datetime,
) -> dict[str, int]:
    stmt = (
        select(UsageEvent.event_type, func.count())
        .where(
            UsageEvent.organization_id == organization_id,
            UsageEvent.occurred_at >= period_start,
            UsageEvent.occurred_at < period_end,
        )
        .group_by(UsageEvent.event_type)
    )
    rows = (await db.execute(stmt)).all()
    return {row[0].value: int(row[1]) for row in rows}


async def record_event(
    db: AsyncSession,
    *,
    organization_id: str,
    user_id: Optional[str],
    event_type: UsageEventType,
    request_id: Optional[str] = None,
    quantity: int = 1,
) -> None:
    db.add(
        UsageEvent(
            organization_id=organization_id,
            user_id=user_id,
            event_type=event_type,
            quantity=quantity,
            request_id=request_id,
        )
    )
    # Caller's session commits on request end (see backend.app.core.db.get_db).


async def current_period_window(subscription: Subscription) -> tuple[datetime, datetime]:
    return subscription.current_period_start, subscription.current_period_end

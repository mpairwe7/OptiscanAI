"""Async SQLAlchemy engine + session factory.

When `settings.database.enabled` is False the engine is not initialized and
`get_db` raises HTTPException(503) — billing/auth endpoints become unavailable
but the rest of the API continues to work, preserving the on-prem/research
deployment story.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def init_engine() -> None:
    """Create the async engine. Idempotent."""
    global _engine, _SessionLocal
    if _engine is not None:
        return
    if not settings.database.enabled:
        logger.info("Database disabled — skipping engine init")
        return
    _engine = create_async_engine(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=settings.database.pool_pre_ping,
        echo=settings.database.echo,
        future=True,
    )
    _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    logger.info("Database engine initialized")


async def dispose_engine() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database engine disposed")


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — yields one AsyncSession per request."""
    if _SessionLocal is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized. Set DATABASE__ENABLED=true.",
        )
    async with _SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


def get_engine() -> Optional[AsyncEngine]:
    return _engine


def session_factory() -> Optional[async_sessionmaker[AsyncSession]]:
    """Direct access to the session factory for code paths outside FastAPI deps
    (e.g. background tasks, manual quota checks inside non-billing endpoints)."""
    return _SessionLocal

"""Async SQLAlchemy engine."""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..config import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_settings().database_url, echo=False)
    return _engine

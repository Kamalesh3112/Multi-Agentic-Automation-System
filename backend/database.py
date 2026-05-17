from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import get_settings


LOGGER = logging.getLogger("backend.database")
SETTINGS = get_settings()

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_init_lock = asyncio.Lock()


def _get_database_url() -> str:
    database_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_DSN")
        or os.getenv("POSTGRES_URL")
        or os.getenv("ASYNC_DATABASE_URL")
    )
    if database_url:
        normalized = database_url.strip()
        if normalized.startswith("postgresql://"):
            return normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
        if normalized.startswith("postgres://"):
            return normalized.replace("postgres://", "postgresql+asyncpg://", 1)
        return normalized

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "app")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


def _json_log(event: str, **payload: object) -> None:
    LOGGER.info(event, extra={"event": event, **payload})


def _build_engine() -> AsyncEngine:
    database_url = _get_database_url()
    pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow = int(os.getenv("DB_POOL_MAX_OVERFLOW", "20"))
    pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
    statement_timeout_ms = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "30000"))

    return create_async_engine(
        database_url,
        echo=SETTINGS.debug,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        connect_args={
            "timeout": connect_timeout,
            "server_settings": {"statement_timeout": str(statement_timeout_ms)},
        },
    )


async def initialize_database(max_retries: int = 5, base_delay_seconds: float = 1.0) -> None:
    global _engine, _session_factory

    if _engine is not None and _session_factory is not None:
        return

    async with _init_lock:
        if _engine is not None and _session_factory is not None:
            return

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                engine = _build_engine()
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))

                _engine = engine
                _session_factory = async_sessionmaker(
                    bind=_engine,
                    expire_on_commit=False,
                    class_=AsyncSession,
                    autoflush=False,
                )
                _json_log("database_initialized", attempt=attempt)
                return
            except (SQLAlchemyError, OSError, ValueError) as exc:
                last_error = exc
                delay = base_delay_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "database_init_retry",
                    extra={"attempt": attempt, "max_retries": max_retries, "delay_seconds": delay},
                )
                if attempt >= max_retries:
                    break
                await asyncio.sleep(delay)

        LOGGER.exception("database_init_failed", extra={"max_retries": max_retries})
        raise RuntimeError("Database initialization failed after retries") from last_error


async def shutdown_database() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _json_log("database_disposed")
    _engine = None
    _session_factory = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine has not been initialized. Call initialize_database() at startup.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Session factory has not been initialized. Call initialize_database() at startup.")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    session = get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with session_scope() as session:
        yield session


def register_database_lifecycle(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup_database() -> None:
        await initialize_database()

    @app.on_event("shutdown")
    async def _shutdown_database() -> None:
        await shutdown_database()

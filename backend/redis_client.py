from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from fastapi import FastAPI

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from backend.config import get_settings


LOGGER = logging.getLogger("backend.redis")
SETTINGS = get_settings()

_pool: ConnectionPool | None = None
_redis: Redis | None = None
_init_lock = asyncio.Lock()


def _redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    password = os.getenv("REDIS_PASSWORD")
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def _build_pool() -> ConnectionPool:
    return ConnectionPool.from_url(
        _redis_url(),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100")),
        socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
        retry_on_timeout=True,
        decode_responses=True,
    )


async def initialize_redis(max_retries: int = 5, base_delay_seconds: float = 1.0) -> None:
    global _pool, _redis

    if _pool is not None and _redis is not None:
        return

    async with _init_lock:
        if _pool is not None and _redis is not None:
            return

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            pool: ConnectionPool | None = None
            redis_client: Redis | None = None
            try:
                pool = _build_pool()
                redis_client = Redis(connection_pool=pool)
                await redis_client.ping()
                _pool = pool
                _redis = redis_client
                LOGGER.info("redis_initialized", extra={"attempt": attempt})
                return
            except (RedisError, OSError, ValueError) as exc:
                last_error = exc
                delay = base_delay_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "redis_init_retry",
                    extra={"attempt": attempt, "max_retries": max_retries, "delay_seconds": delay},
                )
                if redis_client is not None:
                    await redis_client.aclose()
                elif pool is not None:
                    await pool.aclose()
                if attempt >= max_retries:
                    break
                await asyncio.sleep(delay)

        LOGGER.exception("redis_init_failed", extra={"max_retries": max_retries})
        raise RuntimeError("Redis initialization failed after retries") from last_error


async def shutdown_redis() -> None:
    global _pool, _redis
    if _redis is not None:
        await _redis.aclose()
        LOGGER.info("redis_client_closed")
    elif _pool is not None:
        await _pool.aclose()
        LOGGER.info("redis_pool_closed")

    _redis = None
    _pool = None


def get_redis_client() -> Redis:
    if _redis is None:
        raise RuntimeError("Redis client has not been initialized. Call initialize_redis() at startup.")
    return _redis


async def redis_healthcheck() -> dict[str, Any]:
    client = get_redis_client()
    pong = await client.ping()
    return {"redis": "ok" if pong else "degraded"}


def register_redis_lifecycle(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup_redis() -> None:
        await initialize_redis()

    @app.on_event("shutdown")
    async def _shutdown_redis() -> None:
        await shutdown_redis()

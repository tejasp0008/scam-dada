"""Health and readiness.

`/healthz` answers whether the process is up. `/readyz` answers whether its dependencies
are, and reports which provider adapters are wired, so a demo-day misconfiguration (mock
TTS still selected) is visible in one request instead of discovered on stage.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import Resources

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str
    app_env: str


class Readiness(BaseModel):
    ready: bool
    postgres: bool
    redis: bool
    providers: dict[str, str]


@router.get("/healthz", response_model=Health)
async def healthz(resources: Resources) -> Health:
    """Liveness: the process is serving."""
    return Health(status="ok", app_env=resources.settings.app_env)


@router.get("/readyz", response_model=Readiness)
async def readyz(resources: Resources, response: Response) -> Readiness:
    """Readiness: datastores reachable, and which providers are selected."""
    postgres_ok = False
    redis_ok = False
    try:
        async with resources.session_factory() as session:
            await session.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        logger.warning("postgres not ready", exc_info=True)
    try:
        await resources.redis.ping()
        redis_ok = True
    except Exception:
        logger.warning("redis not ready", exc_info=True)

    ready = postgres_ok and redis_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Readiness(
        ready=ready,
        postgres=postgres_ok,
        redis=redis_ok,
        providers={
            "stt": resources.stt.name,
            "tts": resources.tts.name,
            "llm": resources.llm.name,
        },
    )

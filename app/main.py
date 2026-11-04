"""Application factory and lifespan.

Everything expensive is built once here and handed to routes through `app.state`, so there
are no module-level singletons other than settings.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import AppResources
from app.api.routes import health
from app.infra.budget import BudgetGuard
from app.infra.db import create_engine, create_session_factory
from app.infra.logging import configure_logging
from app.infra.redis_bus import EventBus, create_redis
from app.infra.settings import Settings, get_settings
from app.providers.factory import build_llm_provider, build_stt_provider, build_tts_provider

logger = logging.getLogger(__name__)


def build_resources(settings: Settings) -> AppResources:
    """Construct every long-lived object the app needs."""
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    redis = create_redis(settings)
    return AppResources(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        redis=redis,
        bus=EventBus(redis),
        budget=BudgetGuard(redis, session_factory, settings),
        stt=build_stt_provider(settings.stt_provider),
        tts=build_tts_provider(settings.tts_provider),
        llm=build_llm_provider(settings.llm_provider),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build resources on startup, dispose them on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    resources = build_resources(settings)
    app.state.resources = resources
    logger.info(
        "application started",
        extra={
            "app_env": settings.app_env,
            "stt": resources.stt.name,
            "tts": resources.tts.name,
            "llm": resources.llm.name,
        },
    )
    try:
        yield
    finally:
        await resources.redis.aclose()
        await resources.engine.dispose()
        logger.info("application stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Tests pass their own settings instead of mutating the environment."""
    resolved = settings or get_settings()
    app = FastAPI(
        title="Scam Dadi",
        version="0.1.0",
        description="AI scambaiting for Indian phone scams.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(health.router)
    return app


app = create_app()

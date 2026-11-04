"""Application factory and lifespan.

Everything expensive is built once here and handed to routes through `app.state`, so there
are no module-level singletons other than settings.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.deps import AppResources
from app.api.routes import health
from app.infra.budget import BudgetGuard
from app.infra.db import create_engine, create_session_factory
from app.infra.logging import configure_logging
from app.infra.redis_bus import EventBus, create_redis
from app.infra.settings import Settings, get_settings
from app.infra.tts_cache import CachingTTS
from app.providers.factory import build_llm_provider, build_stt_provider, build_tts_provider

logger = logging.getLogger(__name__)


def build_resources(settings: Settings) -> AppResources:
    """Construct every long-lived object the app needs.

    The budget guard is built before the providers because it is their usage recorder: an
    adapter prices what it is about to spend through the guard and never holds a rate.
    """
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    redis = create_redis(settings)
    budget = BudgetGuard(redis, session_factory, settings)
    # One client for every outbound HTTP provider, so connections are pooled and reused:
    # a fresh TLS handshake per TTS request would show up directly as persona latency.
    http = httpx.AsyncClient()

    return AppResources(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        redis=redis,
        http=http,
        bus=EventBus(redis),
        budget=budget,
        stt=build_stt_provider(
            settings.stt_provider,
            api_key=settings.gnani_api_key,
            ws_url=settings.gnani_stt_ws_url,
            model=settings.stt_model,
            usage=budget,
        ),
        tts=CachingTTS(
            build_tts_provider(
                settings.tts_provider,
                client=http,
                api_key=settings.gnani_api_key,
                sse_url=settings.gnani_tts_sse_url,
                model=settings.tts_model,
                language=settings.tts_language,
                usage=budget,
            ),
            settings.tts_cache_dir,
            usage=budget,
        ),
        llm=build_llm_provider(
            settings.llm_provider,
            client=http,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            fallback_model=settings.llm_fallback_model,
            fallback_reasoning_effort=settings.llm_fallback_reasoning_effort,
            first_token_timeout=settings.llm_first_token_timeout_s,
            request_timeout=settings.llm_request_timeout_s,
            usage=budget,
        ),
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
        await resources.http.aclose()
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

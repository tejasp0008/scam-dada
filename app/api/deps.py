"""Dependency wiring: the composition root.

Long-lived objects (engine, Redis client, providers) are built once in the lifespan handler
and stored on `app.state`; these functions hand them to routes. Nothing else constructs a
provider or an engine, which is what keeps the layering in the brief honest.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.infra.budget import BudgetGuard
from app.infra.db import session_scope
from app.infra.redis_bus import EventBus
from app.infra.settings import Settings
from app.providers.base import LLMProvider, STTProvider, TTSProvider


@dataclass(slots=True)
class AppResources:
    """Everything built at startup and torn down at shutdown."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    bus: EventBus
    budget: BudgetGuard
    stt: STTProvider
    tts: TTSProvider
    llm: LLMProvider


def get_resources(request: Request) -> AppResources:
    """Return the resources bundle created by the lifespan handler."""
    resources: AppResources = request.app.state.resources
    return resources


Resources = Annotated[AppResources, Depends(get_resources)]


def get_settings_dep(resources: Resources) -> Settings:
    return resources.settings


def get_bus(resources: Resources) -> EventBus:
    return resources.bus


def get_budget(resources: Resources) -> BudgetGuard:
    return resources.budget


async def get_session(resources: Resources) -> AsyncIterator[AsyncSession]:
    """Yield a transactional database session for the duration of the request."""
    async for session in session_scope(resources.session_factory):
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
BusDep = Annotated[EventBus, Depends(get_bus)]
BudgetDep = Annotated[BudgetGuard, Depends(get_budget)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

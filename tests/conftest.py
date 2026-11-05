"""Shared fixtures.

Every test here runs offline: providers are the mocks and the database is in-memory
SQLite, so `pytest` needs no Docker and spends nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.models_sql import Base
from app.infra.settings import Settings
from app.main import create_app
from app.providers.factory import ProviderName


@pytest.fixture
def settings() -> Settings:
    """Settings pinned to the mock providers, independent of any local .env."""
    return Settings(
        app_env="test",
        log_level="WARNING",
        stt_provider=ProviderName.MOCK,
        tts_provider=ProviderName.MOCK,
        llm_provider=ProviderName.MOCK,
        # No accounting in tests. Mock providers would otherwise write zero-credit rows into
        # the same table that backs every credit figure in the README.
        budget_enabled=False,
        _env_file=None,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A test client with the lifespan handler run, so app.state.resources exists."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
async def sqlite_sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """An in-memory database with the schema created from the ORM metadata.

    Built from metadata rather than by running Alembic, because the baseline migration
    targets Postgres (JSONB). The migration is verified separately against real Postgres.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
async def call_client(
    settings: Settings, sqlite_sessions: async_sessionmaker[AsyncSession]
) -> AsyncIterator[TestClient]:
    """A client whose persona voices are configured and whose writes go to SQLite.

    The session factory is swapped after startup rather than configured, so the test never
    touches the development database even when Docker happens to be running.
    """
    configured = settings.model_copy(
        update={"persona_elder_voice": "Devansh", "persona_grandson_voice": "Devansh"}
    )
    with TestClient(create_app(configured)) as test_client:
        test_client.app.state.resources.session_factory = sqlite_sessions  # type: ignore[attr-defined]
        yield test_client

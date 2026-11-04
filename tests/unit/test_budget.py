"""The budget guard.

The credit pool is finite and the contest has no second chance, so the guard is tested
with a fake Redis rather than left to be discovered on demo day.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infra.budget import BudgetGuard, usage_key
from app.infra.models_sql import ApiUsage
from app.infra.settings import Settings
from app.providers.base import BudgetExceeded, UnitType


class FakeRedis:
    """The two operations the guard uses."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        value = self.store.get(key)
        return None if value is None else str(value).encode()

    async def incrby(self, key: str, amount: int) -> int:
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "budget_enabled": True,
        "budget_max_tts_chars": 100,
        "budget_max_stt_seconds": 60,
        "budget_max_llm_tokens": 1000,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def guard(sqlite_sessions: async_sessionmaker[AsyncSession]) -> BudgetGuard:
    return BudgetGuard(FakeRedis(), sqlite_sessions, make_settings())  # type: ignore[arg-type]


async def test_a_call_within_the_ceiling_is_allowed(guard: BudgetGuard) -> None:
    await guard.check("tts", UnitType.TTS_CHARS, 100)


async def test_a_call_over_the_ceiling_is_refused(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetExceeded, match="tts_chars"):
        await guard.check("tts", UnitType.TTS_CHARS, 101)


async def test_the_ceiling_counts_what_was_already_spent(guard: BudgetGuard) -> None:
    await guard.record("tts", UnitType.TTS_CHARS, 60)
    assert await guard.consumed(UnitType.TTS_CHARS) == 60
    assert await guard.remaining(UnitType.TTS_CHARS) == 40
    await guard.check("tts", UnitType.TTS_CHARS, 40)
    with pytest.raises(BudgetExceeded):
        await guard.check("tts", UnitType.TTS_CHARS, 41)


async def test_unit_types_have_independent_ceilings(guard: BudgetGuard) -> None:
    await guard.record("tts", UnitType.TTS_CHARS, 100)
    with pytest.raises(BudgetExceeded):
        await guard.check("tts", UnitType.TTS_CHARS, 1)
    await guard.check("stt", UnitType.STT_SECONDS, 60)


async def test_a_cache_hit_costs_nothing_but_is_still_recorded(guard: BudgetGuard) -> None:
    """Recording zero-unit hits is what makes "credits saved" a measured number."""
    await guard.record("tts", UnitType.TTS_CHARS, 500, cache_hit=True)
    assert await guard.consumed(UnitType.TTS_CHARS) == 0
    await guard.check("tts", UnitType.TTS_CHARS, 100)


async def test_every_call_is_persisted(
    guard: BudgetGuard, sqlite_sessions: async_sessionmaker[AsyncSession]
) -> None:
    await guard.record("tts", UnitType.TTS_CHARS, 40)
    await guard.record("tts", UnitType.TTS_CHARS, 40, cache_hit=True)
    async with sqlite_sessions() as session:
        rows = (await session.execute(select(ApiUsage))).scalars().all()
        total = (await session.execute(select(func.sum(ApiUsage.units)))).scalar_one()
    assert len(rows) == 2
    assert total == 40
    assert {row.cache_hit for row in rows} == {True, False}


async def test_a_disabled_guard_never_refuses(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard = BudgetGuard(FakeRedis(), sqlite_sessions, make_settings(budget_enabled=False))  # type: ignore[arg-type]
    await guard.check("tts", UnitType.TTS_CHARS, 10_000_000)


def test_counter_keys_are_namespaced_per_unit() -> None:
    keys = {usage_key(unit) for unit in UnitType}
    assert len(keys) == len(UnitType)
    assert all(key.startswith("budget:units:") for key in keys)

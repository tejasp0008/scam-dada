"""The budget guard.

The credit pool is finite and the contest has no second chance, so the guard is tested
with a fake Redis rather than left to be discovered on demo day.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from app.core.cost import BillingPool, CostModel, UnitType
from app.infra.budget import BudgetGuard, credits_key, units_key
from app.infra.models_sql import ApiUsage
from app.infra.settings import Settings
from app.providers.base import BudgetExceeded


class FakePipeline:
    """Queues commands synchronously and applies them on execute, like redis-py."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._queued: list[tuple[str, float]] = []

    def incrbyfloat(self, key: str, amount: float) -> FakePipeline:
        self._queued.append((key, amount))
        return self

    async def execute(self) -> None:
        for key, amount in self._queued:
            await self._redis.incrbyfloat(key, amount)
        self._queued.clear()


class FakeRedis:
    """The operations the guard uses."""

    def __init__(self) -> None:
        self.store: dict[str, float] = {}

    async def get(self, key: str) -> bytes | None:
        value = self.store.get(key)
        return None if value is None else str(value).encode()

    async def incrbyfloat(self, key: str, amount: float) -> float:
        self.store[key] = self.store.get(key, 0.0) + amount
        return self.store[key]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "budget_enabled": True,
        "gnani_credit_total": 100.0,
        "gnani_credit_stop_at": 90.0,
        "llm_credit_total": None,
        "llm_credit_stop_at": None,
        "budget_max_tts_chars": None,
        "budget_max_stt_seconds": None,
        "budget_max_llm_tokens": None,
        "rates_confirmed": True,
        "stt_rate_per_minute": 1.0,
        "tts_rate_per_1k_chars": 1000.0,  # ₹1 per character, so sums are easy to read
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_guard(
    sessions: async_sessionmaker[AsyncSession], **overrides: object
) -> tuple[BudgetGuard, FakeRedis]:
    redis = FakeRedis()
    settings = make_settings(**overrides)
    return BudgetGuard(redis, sessions, settings, settings.cost_model()), redis  # type: ignore[arg-type]


@pytest.fixture
def guard(sqlite_sessions: async_sessionmaker[AsyncSession]) -> BudgetGuard:
    return make_guard(sqlite_sessions)[0]


@pytest.fixture
def disabled_guard(sqlite_sessions: async_sessionmaker[AsyncSession]) -> BudgetGuard:
    return make_guard(sqlite_sessions, budget_enabled=False)[0]


async def rows(guard: BudgetGuard) -> int:
    """How many usage rows the guard has written."""
    async with guard._session_factory() as session:
        result = await session.execute(select(func.count()).select_from(ApiUsage))
        return int(result.scalar_one())


async def test_a_call_within_the_ceiling_is_allowed(guard: BudgetGuard) -> None:
    await guard.check("tts", guard.cost.price_tts(50))


async def test_a_call_over_the_hard_stop_is_refused(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetExceeded, match="credit ceiling"):
        await guard.check("tts", guard.cost.price_tts(91))


async def test_the_hard_stop_counts_what_was_already_spent(guard: BudgetGuard) -> None:
    await guard.record("tts", guard.cost.price_tts(60))
    assert await guard.spent_credits(BillingPool.GNANI) == pytest.approx(60.0)
    await guard.check("tts", guard.cost.price_tts(30))
    with pytest.raises(BudgetExceeded):
        await guard.check("tts", guard.cost.price_tts(31))


async def test_the_hard_stop_sits_below_the_balance(guard: BudgetGuard) -> None:
    """The margin is the point: it leaves credit to re-record the demo."""
    status = await guard.status(BillingPool.GNANI)
    assert status.total == 100.0
    assert status.stop_at == 90.0
    assert status.remaining == 90.0


async def test_the_llm_pool_is_independent_of_gnani_credits(guard: BudgetGuard) -> None:
    """Groq spend must never consume the Gnani ceiling, or a real limit gets hidden."""
    await guard.record("llm", guard.cost.price_llm(1_000_000))
    assert await guard.spent_credits(BillingPool.GNANI) == 0.0
    await guard.check("tts", guard.cost.price_tts(90))


async def test_stt_is_charged_on_billed_minutes_not_raw_seconds(guard: BudgetGuard) -> None:
    """At ₹1/minute, a 2.7 s call costs ₹0.05 because it bills as 0.05 minutes."""
    charge = guard.cost.price_stt(2.7)
    await guard.record("stt", charge)
    assert await guard.spent_credits(BillingPool.GNANI) == pytest.approx(0.05)
    assert await guard.consumed_units(UnitType.STT_SECONDS) == pytest.approx(2.7)


async def test_raw_units_are_recorded_alongside_credits(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """So the rate conversion can be verified, and corrected retroactively if wrong."""
    guard, _ = make_guard(sqlite_sessions)
    await guard.record("stt", guard.cost.price_stt(2.7))
    async with sqlite_sessions() as session:
        row = (await session.execute(select(ApiUsage))).scalars().one()
    assert row.raw_unit == "stt_seconds"
    assert row.raw_units == pytest.approx(2.7)
    assert row.billed_unit == "stt_minutes"
    assert row.billed_units == pytest.approx(0.05)
    assert row.credits == pytest.approx(0.05)
    assert row.pool == "gnani"


async def test_a_raw_unit_ceiling_still_applies_when_no_rate_is_known(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions, llm_rate_per_1m_tokens=0.0, budget_max_llm_tokens=1000)
    await guard.check("llm", guard.cost.price_llm(1000))
    with pytest.raises(BudgetExceeded, match="raw unit ceiling"):
        await guard.check("llm", guard.cost.price_llm(1001))


async def test_a_cache_hit_costs_nothing_and_records_what_it_avoided(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions)
    await guard.record("tts", guard.cost.price_tts(50), cache_hit=True)
    assert await guard.spent_credits(BillingPool.GNANI) == 0.0
    assert await guard.avoided_credits() == pytest.approx(50.0)


async def test_every_call_is_persisted(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions)
    await guard.record("tts", guard.cost.price_tts(40))
    await guard.record("tts", guard.cost.price_tts(40), cache_hit=True)
    async with sqlite_sessions() as session:
        rows = (await session.execute(select(ApiUsage))).scalars().all()
        spent = (await session.execute(select(func.sum(ApiUsage.credits)))).scalar_one()
    assert len(rows) == 2
    assert spent == pytest.approx(40.0)
    assert {row.cache_hit for row in rows} == {True, False}


async def test_estimated_rates_are_flagged_on_the_stored_row(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions, rates_confirmed=False)
    await guard.record("tts", guard.cost.price_tts(10))
    async with sqlite_sessions() as session:
        row = (await session.execute(select(ApiUsage))).scalars().one()
    assert row.credits_estimated is True
    assert (await guard.status(BillingPool.GNANI)).estimated is True


async def test_a_disabled_guard_never_refuses(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions, budget_enabled=False)
    await guard.check("tts", guard.cost.price_tts(10_000_000))


async def test_a_pool_with_no_configured_stop_is_unlimited(
    sqlite_sessions: async_sessionmaker[AsyncSession],
) -> None:
    guard, _ = make_guard(sqlite_sessions, gnani_credit_stop_at=None)
    await guard.check("tts", guard.cost.price_tts(10_000_000))
    assert (await guard.status(BillingPool.GNANI)).remaining is None


async def test_the_default_settings_use_the_programme_balance() -> None:
    """5,000 credits granted, hard stop at 4,500 used, as specified."""
    settings = Settings(_env_file=None)
    assert settings.gnani_credit_total == 5_000.0
    assert settings.gnani_credit_stop_at == 4_500.0
    assert settings.rates_confirmed is False


def test_counter_keys_are_namespaced() -> None:
    assert len({credits_key(p) for p in BillingPool}) == len(BillingPool)
    assert len({units_key(u) for u in UnitType}) == len(UnitType)
    assert all(credits_key(p).startswith("budget:credits:") for p in BillingPool)


def test_the_cost_model_is_reachable_from_the_guard(guard: BudgetGuard) -> None:
    """Adapters price through the guard, so they never hold a rate themselves."""
    assert isinstance(guard.cost, CostModel)


class TestDisabledGuard:
    """`BUDGET_ENABLED=false` means no accounting, not merely no enforcement.

    Tests run with mock providers against whatever database is configured. A guard that
    still wrote rows would put invented usage into the table that backs every credit figure
    in the README, and a ledger with invented rows in it is not a measurement.
    """

    async def test_it_records_nothing(self, disabled_guard: BudgetGuard) -> None:
        await disabled_guard.record(
            "mock-tts", disabled_guard.cost.price_tts(1000), call_id="call-1"
        )
        assert await rows(disabled_guard) == 0

    async def test_it_enforces_nothing(self, disabled_guard: BudgetGuard) -> None:
        charge = disabled_guard.cost.price_tts(10_000_000)
        await disabled_guard.check("mock-tts", charge)  # would raise if enforced

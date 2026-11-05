"""The budget guard: what has been spent, and whether the next call is allowed.

Credits are the thing that actually runs out, so the ceiling is denominated in credits
(₹1 each). Raw units are recorded alongside every charge so the rate conversion stays
auditable — if a rate turns out to be wrong, the credit figures can be re-derived from the
seconds and characters that were actually measured.

Counters live in Redis because the hot path must not wait on Postgres; the durable record
of every call goes to `api_usage`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from app.core.cost import BillingPool, Charge, CostModel, UnitType
from app.infra.models_sql import ApiUsage
from app.infra.settings import Settings
from app.providers.base import BudgetExceeded

logger = logging.getLogger(__name__)


def credits_key(pool: BillingPool) -> str:
    """Redis key holding credits spent from a pool."""
    return f"budget:credits:{pool.value}"


def units_key(unit: UnitType) -> str:
    """Redis key holding raw units consumed of one type."""
    return f"budget:units:{unit.value}"


class BudgetStatus:
    """A snapshot of one pool, for the dashboard and the smoke script."""

    def __init__(
        self,
        pool: BillingPool,
        spent: float,
        stop_at: float | None,
        total: float | None,
        estimated: bool,
    ) -> None:
        self.pool = pool
        self.spent = spent
        self.stop_at = stop_at
        self.total = total
        self.estimated = estimated

    @property
    def remaining(self) -> float | None:
        """Credits left before the hard stop."""
        return None if self.stop_at is None else max(0.0, self.stop_at - self.spent)

    def __repr__(self) -> str:
        total = "unknown" if self.total is None else f"{self.total:g}"
        stop = "none" if self.stop_at is None else f"{self.stop_at:g}"
        flag = " (estimated)" if self.estimated else ""
        return f"<{self.pool.value}: spent {self.spent:.4f} of {total}, hard stop {stop}{flag}>"


class BudgetGuard:
    """Refuses a call that would breach a ceiling, and records what each call cost."""

    def __init__(
        self,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        cost_model: CostModel | None = None,
    ) -> None:
        self._redis = redis
        self._session_factory = session_factory
        self._enabled = settings.budget_enabled
        self._cost = cost_model or settings.cost_model()
        self._totals: dict[BillingPool, float | None] = {
            BillingPool.GNANI: settings.gnani_credit_total,
            BillingPool.LLM: settings.llm_credit_total,
        }
        # A hard stop short of the real balance is the point: it leaves enough credit to
        # re-record the demo if something goes wrong on the day.
        self._stops: dict[BillingPool, float | None] = {
            BillingPool.GNANI: settings.gnani_credit_stop_at,
            BillingPool.LLM: settings.llm_credit_stop_at,
        }
        # Raw-unit ceilings remain as a second line of defence, and are the only ceiling
        # that works at all for a pool whose rate is still unknown.
        self._unit_ceilings: dict[UnitType, int | None] = {
            UnitType.TTS_CHARS: settings.budget_max_tts_chars,
            UnitType.STT_SECONDS: settings.budget_max_stt_seconds,
            UnitType.LLM_TOKENS: settings.budget_max_llm_tokens,
        }

    @property
    def cost(self) -> CostModel:
        """The rates in force."""
        return self._cost

    async def spent_credits(self, pool: BillingPool) -> float:
        """Credits spent from a pool."""
        raw = await self._redis.get(credits_key(pool))
        return float(raw) if raw is not None else 0.0

    async def consumed_units(self, unit: UnitType) -> float:
        """Raw units of one type consumed."""
        raw = await self._redis.get(units_key(unit))
        return float(raw) if raw is not None else 0.0

    async def status(self, pool: BillingPool) -> BudgetStatus:
        """A snapshot of one pool."""
        return BudgetStatus(
            pool=pool,
            spent=await self.spent_credits(pool),
            stop_at=self._stops[pool],
            total=self._totals[pool],
            estimated=not self._cost.rates_confirmed,
        )

    async def check(self, provider: str, charge: Charge) -> None:
        """Raise `BudgetExceeded` if this charge would cross a ceiling."""
        if not self._enabled:
            return

        stop_at = self._stops[charge.pool]
        if stop_at is not None:
            spent = await self.spent_credits(charge.pool)
            if spent + charge.credits > stop_at:
                self._refuse(
                    provider,
                    charge,
                    reason="credit ceiling",
                    detail=f"{spent:.4f} + {charge.credits:.4f} would exceed {stop_at:g}",
                )

        unit_ceiling = self._unit_ceilings[charge.raw_unit]
        if unit_ceiling is not None:
            consumed = await self.consumed_units(charge.raw_unit)
            if consumed + charge.raw_units > unit_ceiling:
                self._refuse(
                    provider,
                    charge,
                    reason="raw unit ceiling",
                    detail=(
                        f"{charge.raw_unit.value}: {consumed:g} + {charge.raw_units:g} "
                        f"would exceed {unit_ceiling}"
                    ),
                )

    def _refuse(self, provider: str, charge: Charge, *, reason: str, detail: str) -> None:
        logger.warning(
            "budget guard refused a call",
            extra={
                "provider": provider,
                "pool": charge.pool.value,
                "reason": reason,
                "detail": detail,
                "credits": charge.credits,
                "estimated_rate": charge.estimated,
            },
        )
        raise BudgetExceeded(f"{reason} for {provider}: {detail}")

    async def record(
        self,
        provider: str,
        charge: Charge,
        *,
        call_id: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Increment the counters (cache hits excepted) and persist a usage row.

        A disabled guard does no accounting at all, not merely no enforcement. That is what
        makes `BUDGET_ENABLED=false` safe in tests: mock providers would otherwise write
        zero-credit rows into the same `api_usage` table that backs every credit figure in
        the README, and a ledger with invented rows in it is not a measurement.
        """
        if not self._enabled:
            return
        if not cache_hit:
            pipe = self._redis.pipeline()
            if charge.credits:
                pipe.incrbyfloat(credits_key(charge.pool), charge.credits)
            if charge.raw_units:
                pipe.incrbyfloat(units_key(charge.raw_unit), charge.raw_units)
            await pipe.execute()

        async with self._session_factory() as session:
            session.add(
                ApiUsage(
                    call_id=call_id,
                    at=datetime.now(tz=UTC),
                    provider=provider,
                    pool=charge.pool.value,
                    raw_unit=charge.raw_unit.value,
                    raw_units=0.0 if cache_hit else charge.raw_units,
                    billed_unit=charge.billed_unit.value,
                    billed_units=0.0 if cache_hit else charge.billed_units,
                    credits=0.0 if cache_hit else charge.credits,
                    credits_estimated=charge.estimated,
                    # What the call would have cost, so caching can be shown to have paid.
                    avoided_credits=charge.credits if cache_hit else 0.0,
                    cache_hit=cache_hit,
                )
            )
            await session.commit()

        logger.info(
            "provider usage",
            extra={
                "provider": provider,
                "pool": charge.pool.value,
                "raw_unit": charge.raw_unit.value,
                "raw_units": charge.raw_units,
                "billed_unit": charge.billed_unit.value,
                "billed_units": 0.0 if cache_hit else charge.billed_units,
                "credits": 0.0 if cache_hit else charge.credits,
                "cache_hit": cache_hit,
                "estimated_rate": charge.estimated,
            },
        )

    async def avoided_credits(self) -> float:
        """Credits not spent because of cache hits, summed from the durable record."""
        async with self._session_factory() as session:
            total = (await session.execute(select(func.sum(ApiUsage.avoided_credits)))).scalar_one()
        return float(total or 0.0)

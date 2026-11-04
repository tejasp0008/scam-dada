"""The budget guard.

Gnani publishes no credit conversion, so this counts raw units per provider and refuses
calls once a configured ceiling is reached. Counters are kept in Redis (the hot path must
not wait on Postgres) and every call is also written to `api_usage` for the honest
end-of-project numbers.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infra.models_sql import ApiUsage
from app.infra.settings import Settings
from app.providers.base import BudgetExceeded, UnitType

logger = logging.getLogger(__name__)


def usage_key(unit: UnitType) -> str:
    """Redis counter key for a unit type."""
    return f"budget:units:{unit.value}"


class BudgetGuard:
    """Checks ceilings before a provider call and records what it cost afterwards."""

    def __init__(
        self,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._redis = redis
        self._session_factory = session_factory
        self._enabled = settings.budget_enabled
        self._ceilings: dict[UnitType, int] = {
            UnitType.TTS_CHARS: settings.budget_max_tts_chars,
            UnitType.STT_SECONDS: settings.budget_max_stt_seconds,
            UnitType.LLM_TOKENS: settings.budget_max_llm_tokens,
        }

    def ceiling(self, unit: UnitType) -> int:
        """The configured ceiling for a unit type."""
        return self._ceilings[unit]

    async def consumed(self, unit: UnitType) -> int:
        """Units spent so far."""
        raw = await self._redis.get(usage_key(unit))
        return int(raw) if raw is not None else 0

    async def remaining(self, unit: UnitType) -> int:
        """Units left before the ceiling."""
        return max(0, self._ceilings[unit] - await self.consumed(unit))

    async def check(self, provider: str, unit: UnitType, units: int) -> None:
        """Raise `BudgetExceeded` if this call would breach the ceiling."""
        if not self._enabled:
            return
        already = await self.consumed(unit)
        ceiling = self._ceilings[unit]
        if already + units > ceiling:
            logger.warning(
                "budget guard refused a call",
                extra={
                    "provider": provider,
                    "unit_type": unit.value,
                    "requested": units,
                    "consumed": already,
                    "ceiling": ceiling,
                },
            )
            raise BudgetExceeded(
                f"{unit.value}: {already} + {units} would exceed the ceiling of {ceiling}"
            )

    async def record(
        self,
        provider: str,
        unit: UnitType,
        units: int,
        *,
        call_id: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Increment the counter (cache hits excepted) and persist a usage row."""
        if not cache_hit and units:
            await self._redis.incrby(usage_key(unit), units)
        async with self._session_factory() as session:
            session.add(
                ApiUsage(
                    call_id=call_id,
                    at=datetime.now(tz=UTC),
                    provider=provider,
                    unit_type=unit.value,
                    units=0 if cache_hit else units,
                    cache_hit=cache_hit,
                )
            )
            await session.commit()
        logger.info(
            "provider usage",
            extra={
                "provider": provider,
                "unit_type": unit.value,
                "units": 0 if cache_hit else units,
                "cache_hit": cache_hit,
            },
        )

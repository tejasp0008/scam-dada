"""The recorder: sequence numbers, and the three places an event has to end up."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.events import StateChangeEvent
from app.core.models import Entity, EntitySource, EntityType, PersonaId, Speaker, Turn
from app.core.state import CallState
from app.infra.models_sql import Call as CallRow
from app.infra.models_sql import Entity as EntityRow
from app.infra.models_sql import Event as EventRow
from app.infra.models_sql import Turn as TurnRow
from app.services.recorder import CallRecorder


class FakeBus:
    def __init__(self, *, broken: bool = False) -> None:
        self.published: list[object] = []
        self._broken = broken

    async def publish(self, event: object) -> None:
        if self._broken:
            raise ConnectionError("redis is not there")
        self.published.append(event)


def state_change(call_id: str, seq: int, ts: datetime) -> StateChangeEvent:
    return StateChangeEvent(
        call_id=call_id,
        seq=seq,
        ts=ts,
        from_state=CallState.RINGING,
        to_state=CallState.HUMAN_CHECK,
        reason="answered",
    )


async def count(sessions: async_sessionmaker[AsyncSession], table: type) -> int:
    async with sessions() as session:
        result = await session.execute(select(func.count()).select_from(table))
        return int(result.scalar_one())


async def seed_call(sessions: async_sessionmaker[AsyncSession], call_id: str) -> None:
    """Turns, events and entities all reference a call row."""
    async with sessions() as session, session.begin():
        session.add(CallRow(id=call_id, state=CallState.RINGING.value, source="test"))


class TestSequencing:
    async def test_sequence_numbers_are_dense_from_zero(self) -> None:
        recorder = CallRecorder("c1", FakeBus())  # type: ignore[arg-type]
        for _ in range(5):
            await recorder.emit(state_change)
        assert [e.seq for e in recorder.history()] == [0, 1, 2, 3, 4]

    async def test_the_caller_cannot_choose_a_sequence_number(self) -> None:
        """Which is what makes gaps and duplicates impossible rather than unlikely."""
        recorder = CallRecorder("c1", FakeBus())  # type: ignore[arg-type]
        event = await recorder.emit(state_change)
        assert event.seq == 0
        assert recorder.next_seq == 1

    async def test_every_event_is_published(self) -> None:
        bus = FakeBus()
        recorder = CallRecorder("c1", bus)  # type: ignore[arg-type]
        await recorder.emit(state_change)
        assert len(bus.published) == 1


class TestSurvivingABrokenBus:
    async def test_a_dead_bus_does_not_end_the_call(self) -> None:
        """A call must not end because the dashboard cannot be fed."""
        recorder = CallRecorder("c1", FakeBus(broken=True))  # type: ignore[arg-type]
        for _ in range(3):
            await recorder.emit(state_change)
        assert recorder.publish_failures == 3
        assert len(recorder.history()) == 3

    async def test_sequence_numbers_still_advance(self) -> None:
        recorder = CallRecorder("c1", FakeBus(broken=True))  # type: ignore[arg-type]
        await recorder.emit(state_change)
        await recorder.emit(state_change)
        assert [e.seq for e in recorder.history()] == [0, 1]


class TestPersistence:
    async def test_events_reach_the_database(
        self, sqlite_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        await seed_call(sqlite_sessions, "c1")
        recorder = CallRecorder("c1", FakeBus(), sqlite_sessions)  # type: ignore[arg-type]
        await recorder.emit(state_change)
        await recorder.drain()
        assert await count(sqlite_sessions, EventRow) == 1

    async def test_turns_reach_the_database(
        self, sqlite_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        await seed_call(sqlite_sessions, "c1")
        recorder = CallRecorder("c1", FakeBus(), sqlite_sessions)  # type: ignore[arg-type]
        recorder.record_turn(
            Turn(
                seq=0,
                speaker=Speaker.PERSONA,
                text="Haan beta",
                persona=PersonaId.DADA,
                started_at_ms=0,
            ),
            latency={"llm_first_token_ms": 300},
        )
        await recorder.drain()
        assert await count(sqlite_sessions, TurnRow) == 1

    async def test_entities_reach_their_own_table(
        self, sqlite_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """The evidence pack asks "every identifier from this call", which is not a
        reasonable question to ask of a JSONB column."""
        await seed_call(sqlite_sessions, "c1")
        recorder = CallRecorder("c1", FakeBus(), sqlite_sessions)  # type: ignore[arg-type]
        recorder.record_entity(
            Entity(
                entity_type=EntityType.UPI_ID,
                value="fake.officer@demo",
                source=EntitySource.REGEX,
                turn_seq=0,
            )
        )
        await recorder.drain()
        assert await count(sqlite_sessions, EntityRow) == 1

    async def test_a_failed_write_does_not_reach_the_caller(self) -> None:
        """A database that refuses every write must cost the evidence pack, not the call.

        The failure is injected rather than provoked with a foreign key: SQLite does not
        enforce them unless asked, so a test that relied on one would pass here for the
        wrong reason and prove nothing about Postgres.
        """

        def broken() -> AsyncSession:
            raise ConnectionError("the database is not there")

        recorder = CallRecorder("c1", FakeBus(), broken)  # type: ignore[arg-type]
        await recorder.emit(state_change)
        recorder.record_entity(
            Entity(
                entity_type=EntityType.PHONE,
                value="9876543210",
                source=EntitySource.REGEX,
                turn_seq=0,
            )
        )
        recorder.record_turn(Turn(seq=0, speaker=Speaker.CALLER, text="x", started_at_ms=0))
        await recorder.drain()
        assert len(recorder.history()) == 1

    async def test_nothing_is_persisted_without_a_session_factory(self) -> None:
        """Used by the unit tests, and by a call whose database is down."""
        recorder = CallRecorder("c1", FakeBus())  # type: ignore[arg-type]
        await recorder.emit(state_change)
        recorder.record_turn(Turn(seq=0, speaker=Speaker.CALLER, text="x", started_at_ms=0))
        await recorder.drain()
        assert len(recorder.history()) == 1


class TestTimestamps:
    async def test_events_are_stamped_in_utc(self) -> None:
        recorder = CallRecorder("c1", FakeBus())  # type: ignore[arg-type]
        event = await recorder.emit(state_change)
        assert event.ts.tzinfo is not None
        assert event.ts.utcoffset() == UTC.utcoffset(None)

    async def test_stamps_do_not_go_backwards(self) -> None:
        recorder = CallRecorder("c1", FakeBus())  # type: ignore[arg-type]
        stamps = [(await recorder.emit(state_change)).ts for _ in range(5)]
        assert stamps == sorted(stamps)


@pytest.mark.parametrize("call_id", ["c1", "another-call"])
async def test_every_event_carries_the_recorders_call_id(call_id: str) -> None:
    recorder = CallRecorder(call_id, FakeBus())  # type: ignore[arg-type]
    event = await recorder.emit(state_change)
    assert event.call_id == call_id

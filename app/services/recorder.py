"""Assigning sequence numbers and getting events to their three destinations.

Every event goes to the dashboard (Redis pub/sub), the replay log (a Redis list) and the
`events` table. The orchestrator must not know that, and must never be slowed by it: a
database round trip on the dialogue path is latency the scammer hears.

So the live fan-out is awaited and the durable write is not. The ordering that matters —
the `seq` — is assigned here, synchronously, before anything is dispatched, so a dashboard
and a replay both see the same order regardless of which write lands first.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime

from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.events import DomainEvent
from app.core.models import Turn as DomainTurn
from app.infra.models_sql import Event as EventRow
from app.infra.models_sql import Turn as TurnRow
from app.infra.redis_bus import EventBus

logger = logging.getLogger(__name__)

_EVENT_ADAPTER: TypeAdapter[DomainEvent] = TypeAdapter(DomainEvent)

#: Takes the fields only the recorder knows - call id, sequence number, timestamp - and
#: returns the finished event. Passing a builder rather than an event is what makes it
#: impossible for a caller to invent its own `seq`.
EventBuilder = Callable[[str, int, datetime], DomainEvent]


class CallRecorder:
    """Emits one call's events, in order, to everything that needs them."""

    def __init__(
        self,
        call_id: str,
        bus: EventBus,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._call_id = call_id
        self._bus = bus
        self._sessions = session_factory
        self._seq = 0
        self._writes: set[asyncio.Task[None]] = set()
        self._publish_failures = 0
        self.emitted: list[DomainEvent] = []

    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def next_seq(self) -> int:
        """The sequence number the next event will get. Exposed for assertions."""
        return self._seq

    def now(self) -> datetime:
        """Timestamp source, in one place so tests can see what was stamped."""
        return datetime.now(UTC)

    async def emit(self, build: EventBuilder) -> DomainEvent:
        """Stamp, publish and persist one event.

        The orchestrator never allocates a sequence number itself, which is what makes gaps
        and duplicates impossible rather than merely unlikely.
        """
        seq = self._seq
        self._seq += 1
        event: DomainEvent = build(self._call_id, seq, self.now())
        self.emitted.append(event)

        await self._publish(event)
        if self._sessions is not None:
            self._schedule(self._persist_event(event))
        return event

    async def _publish(self, event: DomainEvent) -> None:
        """Fan out to live subscribers, surviving a Redis that is not there.

        A call must not end because the dashboard cannot be fed. The event is already in
        `emitted` and is still written to the database, so what is lost is the live view and
        the reconnect log - not the evidence. Logged once per call rather than per event,
        because a Redis outage would otherwise bury every other line in the log.
        """
        try:
            await self._bus.publish(event)
        except Exception:
            self._publish_failures += 1
            if self._publish_failures == 1:
                logger.warning(
                    "publishing events failed; the live dashboard will be blank for this call",
                    extra={"call_id": self._call_id},
                    exc_info=True,
                )

    @property
    def publish_failures(self) -> int:
        """How many events never reached the bus. Zero on a healthy call."""
        return self._publish_failures

    def record_turn(
        self, turn: DomainTurn, *, latency: dict[str, int | None] | None = None
    ) -> None:
        """Persist a completed turn. Fire and forget: nothing on the call path reads it back."""
        if self._sessions is not None:
            self._schedule(self._persist_turn(turn, latency or {}))

    def _schedule(self, coro: Coroutine[object, object, None]) -> None:
        task = asyncio.create_task(coro)
        # Held in a set because asyncio only keeps a weak reference: a task that is not
        # retained can be garbage-collected mid-write.
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _persist_event(self, event: DomainEvent) -> None:
        assert self._sessions is not None
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    EventRow(
                        call_id=event.call_id,
                        seq=event.seq,
                        type=event.type.value,
                        ts=event.ts,
                        payload=_EVENT_ADAPTER.dump_python(event, mode="json"),
                    )
                )
        except Exception:
            # A failed durable write must not end a live call. The dashboard already has the
            # event and the Redis log already holds it; this costs us the evidence pack row.
            logger.exception("persisting an event failed", extra={"call_id": event.call_id})

    async def _persist_turn(self, turn: DomainTurn, latency: dict[str, int | None]) -> None:
        assert self._sessions is not None
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    TurnRow(
                        call_id=self._call_id,
                        seq=turn.seq,
                        speaker=turn.speaker.value,
                        persona=turn.persona.value if turn.persona else None,
                        text=turn.text,
                        language_code=turn.language_code,
                        started_at_ms=turn.started_at_ms,
                        duration_ms=turn.duration_ms,
                        stt_final_ms=latency.get("stt_final_ms"),
                        llm_first_token_ms=latency.get("llm_first_token_ms"),
                        tts_first_byte_ms=latency.get("tts_first_byte_ms"),
                    )
                )
        except Exception:
            logger.exception("persisting a turn failed", extra={"call_id": self._call_id})

    async def drain(self) -> None:
        """Wait for outstanding durable writes. Called once, at the end of a call."""
        if not self._writes:
            return
        await asyncio.gather(*tuple(self._writes), return_exceptions=True)

    def history(self) -> Sequence[DomainEvent]:
        """Everything emitted on this call, in order."""
        return tuple(self.emitted)

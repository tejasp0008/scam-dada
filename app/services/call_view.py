"""Assembling a call report out of the database.

The queries live here; the arithmetic lives in `app.core.report`, where a test can call it
without a database. This module's job is only to fetch rows and hand them to those
functions, which is why it is mostly `select` statements and almost no logic.

One decision worth stating: the summary is recomputed from turns and events rather than read
from denormalised columns on `calls`. The call row does carry `scammer_seconds_wasted`, and
it is written at hang-up, but a live call has not been written yet and a crashed one never
will be. Recomputing means the dashboard shows the same number for a call in progress, a
finished call and a call whose process died - and that number is always derivable from the
evidence rather than from a field somebody remembered to set.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from pydantic import TypeAdapter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import DomainEvent, LatencyEvent, RiskEvent, StageEvent
from app.core.models import Entity, EntitySource, EntityType, PersonaId, ScamStage, Speaker, Turn
from app.core.report import (
    CallReport,
    CallSummary,
    LeakReport,
    count_turns,
    latency_legs,
    personas_in_order,
    seconds_wasted,
)
from app.core.stages import STAGE_ORDER
from app.core.state import CallState
from app.infra.models_sql import Call as CallRow
from app.infra.models_sql import CanaryIdentity as CanaryRow
from app.infra.models_sql import CanaryMatch as MatchRow
from app.infra.models_sql import Entity as EntityRow
from app.infra.models_sql import Event as EventRow
from app.infra.models_sql import Turn as TurnRow

logger = logging.getLogger(__name__)

_EVENT_ADAPTER: TypeAdapter[DomainEvent] = TypeAdapter(DomainEvent)

#: Legs reported in this order, which is the order they happen in a turn.
LATENCY_LEGS = ("stt_final_ms", "llm_first_token_ms", "tts_first_byte_ms")


async def list_calls(session: AsyncSession, *, limit: int = 50) -> list[CallSummary]:
    """Summaries for the most recent calls, newest first."""
    rows = await session.execute(select(CallRow).order_by(CallRow.started_at.desc()).limit(limit))
    calls = list(rows.scalars())
    return [await _summarise(session, row) for row in calls]


async def get_report(session: AsyncSession, call_id: str) -> CallReport | None:
    """The full report for one call, or None if there is no such call."""
    row = (await session.execute(select(CallRow).where(CallRow.id == call_id))).scalar_one_or_none()
    if row is None:
        return None

    turns = await _turns(session, call_id)
    events = await _events(session, call_id)
    summary = await _summarise(session, row, turns=turns, events=events)

    stage_timeline = tuple(
        (event.ts, event.stage) for event in events if isinstance(event, StageEvent)
    )
    tactics = tuple(
        dict.fromkeys(
            event.tactic for event in events if isinstance(event, RiskEvent) and event.tactic
        )
    )
    return CallReport(
        summary=summary,
        turns=turns,
        entities=await _entities(session, call_id),
        leaks=await _leaks(session, call_id),
        latency=latency_legs(_latency_samples(events)),
        canary_issued=await _canary_issued(session, call_id),
        stage_timeline=_dedupe_timeline(stage_timeline),
        tactics_used=tactics,
    )


async def _summarise(
    session: AsyncSession,
    row: CallRow,
    *,
    turns: Sequence[Turn] | None = None,
    events: Sequence[DomainEvent] | None = None,
) -> CallSummary:
    resolved_turns = turns if turns is not None else await _turns(session, row.id)
    resolved_events = events if events is not None else await _events(session, row.id)

    caller_turns, persona_turns = count_turns(resolved_turns)
    stages = [e.stage for e in resolved_events if isinstance(e, StageEvent)]
    pressures = [e.pressure_score for e in resolved_events if isinstance(e, StageEvent)]
    risks = [e.hangup_risk for e in resolved_events if isinstance(e, RiskEvent)]

    ended_ms = None
    if row.ended_at is not None and row.started_at is not None:
        ended_ms = int((row.ended_at - row.started_at).total_seconds() * 1000)

    return CallSummary(
        call_id=row.id,
        started_at=row.started_at,
        ended_at=row.ended_at,
        state=_state(row),
        ended_reason=row.ended_reason,
        source=row.source,
        seconds_wasted=seconds_wasted(resolved_turns, ended_at_ms=ended_ms),
        caller_turns=caller_turns,
        persona_turns=persona_turns,
        personas=personas_in_order(resolved_turns),
        stage_reached=max(stages, key=STAGE_ORDER.index) if stages else None,
        peak_pressure=max(pressures, default=0.0),
        peak_hangup_risk=max(risks, default=0.0),
        entity_count=await _count(session, EntityRow, EntityRow.call_id, row.id),
        leak_count=await _count(session, MatchRow, MatchRow.matched_call_id, row.id),
    )


def _state(row: CallRow) -> CallState:
    """The call's state, tolerating a value written by an older build."""
    for candidate in (row.final_state, row.state):
        if candidate:
            try:
                return CallState(candidate)
            except ValueError:
                logger.info("unrecognised call state", extra={"state": candidate})
    return CallState.ENDED


def _latency_samples(events: Sequence[DomainEvent]) -> dict[str, list[int]]:
    """Per-leg samples, including legs that were never measured.

    Every leg appears even when empty: `stt_final_ms` is often unmeasurable because
    Prisma's own VAD can beat our hysteresis (DECISIONS 0024), and a leg omitted from the
    dashboard reads as fast rather than as absent.
    """
    samples: dict[str, list[int]] = {leg: [] for leg in LATENCY_LEGS}
    for event in events:
        if not isinstance(event, LatencyEvent):
            continue
        for leg in LATENCY_LEGS:
            value = getattr(event, leg)
            if isinstance(value, int):
                samples[leg].append(value)
    return samples


def _dedupe_timeline(
    timeline: Sequence[tuple[datetime, ScamStage]],
) -> tuple[tuple[datetime, ScamStage], ...]:
    """Collapse consecutive repeats, so the timeline marks changes rather than ticks."""
    out: list[tuple[datetime, ScamStage]] = []
    for stamp, stage in timeline:
        if not out or out[-1][1] != stage:
            out.append((stamp, stage))
    return tuple(out)


async def _turns(session: AsyncSession, call_id: str) -> tuple[Turn, ...]:
    rows = await session.execute(
        select(TurnRow).where(TurnRow.call_id == call_id).order_by(TurnRow.seq)
    )
    turns: list[Turn] = []
    for row in rows.scalars():
        try:
            turns.append(
                Turn(
                    seq=row.seq,
                    speaker=Speaker(row.speaker),
                    text=row.text,
                    persona=PersonaId(row.persona) if row.persona else None,
                    language_code=row.language_code,
                    started_at_ms=row.started_at_ms,
                    duration_ms=row.duration_ms,
                )
            )
        except ValueError:
            logger.info("skipping an unparseable turn", extra={"call_id": call_id, "seq": row.seq})
    return tuple(turns)


async def _events(session: AsyncSession, call_id: str) -> tuple[DomainEvent, ...]:
    rows = await session.execute(
        select(EventRow).where(EventRow.call_id == call_id).order_by(EventRow.seq)
    )
    events: list[DomainEvent] = []
    for row in rows.scalars():
        try:
            events.append(_EVENT_ADAPTER.validate_python(row.payload))
        except ValueError:
            # An event written by an older schema. Skipped rather than guessed at: the
            # report is evidence, and a half-understood row is not evidence.
            logger.info("skipping an unparseable event", extra={"call_id": call_id, "seq": row.seq})
    return tuple(events)


async def _entities(session: AsyncSession, call_id: str) -> tuple[Entity, ...]:
    rows = await session.execute(
        select(EntityRow).where(EntityRow.call_id == call_id).order_by(EntityRow.entity_type)
    )
    entities: list[Entity] = []
    for row in rows.scalars():
        try:
            entities.append(
                Entity(
                    entity_type=EntityType(row.entity_type),
                    value=row.value,
                    source=EntitySource(row.source),
                    turn_seq=0,
                    confidence=row.confidence,
                )
            )
        except ValueError:
            logger.info("skipping an unparseable entity", extra={"call_id": call_id})
    return tuple(entities)


async def _leaks(session: AsyncSession, call_id: str) -> tuple[LeakReport, ...]:
    rows = await session.execute(
        select(MatchRow, CanaryRow)
        .join(CanaryRow, CanaryRow.id == MatchRow.identity_id)
        .where(MatchRow.matched_call_id == call_id)
    )
    return tuple(
        LeakReport(
            identity_id=match.identity_id,
            issued_call_id=identity.issued_call_id,
            matched_field=match.matched_field,
            matched_text=match.matched_text,
            score=match.score,
        )
        for match, identity in rows.all()
    )


async def _canary_issued(session: AsyncSession, call_id: str) -> dict[str, str]:
    row = (
        (await session.execute(select(CanaryRow).where(CanaryRow.issued_call_id == call_id)))
        .scalars()
        .first()
    )
    if row is None:
        return {}
    return {key: str(value) for key, value in row.identity.items()}


async def _count(session: AsyncSession, table: type, column: object, call_id: str) -> int:
    result = await session.execute(
        select(func.count()).select_from(table).where(column == call_id)  # type: ignore[arg-type]
    )
    return int(result.scalar_one())

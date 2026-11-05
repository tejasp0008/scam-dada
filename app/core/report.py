"""The shape of a finished call, and the arithmetic that summarises one.

Three consumers read this: the control-room dashboard, the REST API, and the evidence pack.
They get the same objects, computed the same way, because a dashboard that disagrees with
the PDF it generated is worse than either on its own.

Everything here is pure. The numbers that will be shown to a judge - how long a scammer was
kept on the line, how many identifiers were extracted, the median latency per leg - are
computed from turns and events by functions a test can call directly. That is the whole
reason this is not a SQL query: "no hardcoded results" means a number has to come from a
real run, and it also has to be reproducible from that run's data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import Entity, PersonaId, ScamStage, Speaker, Turn
from app.core.state import CallState


class LatencyLeg(BaseModel):
    """One leg's latency distribution across a call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    samples: int = Field(ge=0)
    median_ms: int | None = None
    p95_ms: int | None = None
    worst_ms: int | None = None


class LeakReport(BaseModel):
    """A canary from an earlier call, found in this one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str
    issued_call_id: str
    matched_field: str
    matched_text: str
    score: float = Field(ge=0.0, le=1.0)


class CallSummary(BaseModel):
    """The one-line view of a call, as the dashboard list shows it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    started_at: datetime
    ended_at: datetime | None = None
    state: CallState
    ended_reason: str | None = None
    source: str = "browser"

    #: The headline number. Seconds of a scammer's working day that this call consumed.
    seconds_wasted: float = Field(default=0.0, ge=0.0)
    caller_turns: int = Field(default=0, ge=0)
    persona_turns: int = Field(default=0, ge=0)
    personas: tuple[PersonaId, ...] = ()
    stage_reached: ScamStage | None = None
    peak_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    peak_hangup_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_count: int = Field(default=0, ge=0)
    leak_count: int = Field(default=0, ge=0)
    # There is deliberately no `barge_ins` here. Barge-in happens, is logged, and is
    # visible live in the simulator, but it is not an event and not a column, so a field on
    # the summary could only ever have reported zero. A number that is always zero is worse
    # than an absent one: it reads as "never happened". Emitting a barge-in event is the
    # right fix and is not done yet.

    @property
    def minutes_wasted(self) -> float:
        return round(self.seconds_wasted / 60.0, 2)

    @property
    def is_live(self) -> bool:
        return self.ended_at is None


class CallReport(BaseModel):
    """Everything about one call. The evidence pack is a rendering of this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: CallSummary
    turns: tuple[Turn, ...] = ()
    entities: tuple[Entity, ...] = ()
    leaks: tuple[LeakReport, ...] = ()
    latency: tuple[LatencyLeg, ...] = ()
    canary_issued: dict[str, str] = Field(
        default_factory=dict,
        description="The fictional identity this call was prepared to hand out.",
    )
    stage_timeline: tuple[tuple[datetime, ScamStage], ...] = ()
    tactics_used: tuple[str, ...] = ()
    notes: str = ""

    @property
    def entities_by_type(self) -> dict[str, list[Entity]]:
        """Grouped for display, in a stable order."""
        grouped: dict[str, list[Entity]] = {}
        for entity in self.entities:
            grouped.setdefault(entity.entity_type.value, []).append(entity)
        return dict(sorted(grouped.items()))


def percentile(values: Sequence[int], fraction: float) -> int | None:
    """Nearest-rank percentile. None for an empty sample.

    Nearest-rank rather than interpolated: with the handful of turns a demo call produces,
    an interpolated p95 invents a number that no turn actually took, and every figure here
    has to correspond to something that happened.
    """
    if not values:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def latency_legs(samples: Mapping[str, Sequence[int]]) -> tuple[LatencyLeg, ...]:
    """Summarise each latency leg, keeping the sample size visible.

    `samples` is empty-tolerant: a leg with no measurements reports `samples=0` and no
    numbers rather than being omitted, because a missing leg in the dashboard reads as
    "fast" when it means "never measured". `stt_final_ms` is often unmeasurable - see
    DECISIONS 0024.
    """
    return tuple(
        LatencyLeg(
            name=name,
            samples=len(values),
            median_ms=percentile(values, 0.5),
            p95_ms=percentile(values, 0.95),
            worst_ms=max(values) if values else None,
        )
        for name, values in samples.items()
    )


def count_turns(turns: Sequence[Turn]) -> tuple[int, int]:
    """(caller turns, persona turns)."""
    caller = sum(1 for turn in turns if turn.speaker is Speaker.CALLER)
    persona = sum(1 for turn in turns if turn.speaker is Speaker.PERSONA)
    return caller, persona


def personas_in_order(turns: Sequence[Turn]) -> tuple[PersonaId, ...]:
    """Who held the phone, in the order they first spoke."""
    seen: list[PersonaId] = []
    for turn in turns:
        if turn.speaker is Speaker.PERSONA and turn.persona and turn.persona not in seen:
            seen.append(turn.persona)
    return tuple(seen)


def seconds_wasted(turns: Sequence[Turn], *, ended_at_ms: int | None = None) -> float:
    """How long the call held the caller, in seconds.

    Measured from the first turn to the last, or to `ended_at_ms` when the call's own end is
    known. Deliberately *not* the sum of turn durations: the silences between turns are the
    scammer's time too, and they are most of what a stalling persona produces.
    """
    if not turns:
        return 0.0
    start = min(turn.started_at_ms for turn in turns)
    last = max(turn.started_at_ms + (turn.duration_ms or 0) for turn in turns)
    end = max(last, ended_at_ms) if ended_at_ms is not None else last
    return round(max(0, end - start) / 1000.0, 2)

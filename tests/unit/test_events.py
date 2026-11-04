"""The event schema.

These are contract tests in all but name: the same JSON goes to the dashboard, into JSONB
and into the evidence pack, so the union must round-trip losslessly and must refuse
anything it does not recognise.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.events import (
    CanaryMatchEvent,
    DomainEvent,
    EntityEvent,
    EventType,
    LatencyEvent,
    PersonaHandoffEvent,
    RiskEvent,
    StageEvent,
    StateChangeEvent,
    TranscriptEvent,
)
from app.core.models import EntitySource, EntityType, PersonaId, ScamStage, Speaker
from app.core.state import CallState

ADAPTER: TypeAdapter[DomainEvent] = TypeAdapter(DomainEvent)
TS = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)

SAMPLES: tuple[DomainEvent, ...] = (
    StateChangeEvent(
        call_id="c1",
        seq=0,
        ts=TS,
        from_state=CallState.HUMAN_CHECK,
        to_state=CallState.PERSONA_ACTIVE,
        reason="barge-in probe answered",
    ),
    TranscriptEvent(
        call_id="c1",
        seq=1,
        ts=TS,
        turn_seq=0,
        speaker=Speaker.CALLER,
        text="main cyber cell se bol raha hoon",
        language_code="hi-IN",
        audio_duration_ms=1800,
    ),
    TranscriptEvent(
        call_id="c1",
        seq=2,
        ts=TS,
        turn_seq=1,
        speaker=Speaker.PERSONA,
        text="haan beta <pauses> kaun?",
        persona=PersonaId.DADI,
    ),
    StageEvent(
        call_id="c1", seq=3, ts=TS, stage=ScamStage.AUTHORITY, confidence=0.8, pressure_score=0.4
    ),
    EntityEvent(
        call_id="c1",
        seq=4,
        ts=TS,
        entity_type=EntityType.UPI_ID,
        value="fake.officer@demo",
        source=EntitySource.REGEX,
        turn_seq=0,
    ),
    RiskEvent(
        call_id="c1",
        seq=5,
        ts=TS,
        hangup_risk=0.7,
        signals=("repetition", "impatience"),
        tactic="otp_arriving",
    ),
    PersonaHandoffEvent(
        call_id="c1",
        seq=6,
        ts=TS,
        from_persona=PersonaId.DADI,
        to_persona=PersonaId.DADA,
        reason="risk high",
    ),
    CanaryMatchEvent(
        call_id="c1",
        seq=7,
        ts=TS,
        identity_id="id-1",
        issued_call_id="c0",
        matched_field="full_name",
        matched_text="kamla devi testwala",
        score=0.92,
    ),
    LatencyEvent(
        call_id="c1",
        seq=8,
        ts=TS,
        turn_seq=1,
        stt_final_ms=210,
        llm_first_token_ms=340,
        tts_first_byte_ms=180,
        filler_played=True,
    ),
)


@pytest.mark.parametrize("event", SAMPLES, ids=lambda e: str(e.type.value))
def test_every_event_round_trips_through_the_union(event: DomainEvent) -> None:
    assert ADAPTER.validate_json(ADAPTER.dump_json(event)) == event


def test_the_union_covers_every_declared_event_type() -> None:
    """A new EventType with no member in the union would fail silently at runtime."""
    assert {e.type for e in SAMPLES} == set(EventType)


def test_the_discriminator_selects_the_right_model() -> None:
    decoded = ADAPTER.validate_python(
        {"type": "risk", "call_id": "c1", "seq": 9, "ts": TS, "hangup_risk": 0.1}
    )
    assert isinstance(decoded, RiskEvent)


def test_events_are_immutable() -> None:
    """Events are an audit trail; a mutable one could be edited after persistence."""
    event = SAMPLES[0]
    with pytest.raises(ValidationError):
        event.seq = 99  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    """A typo'd field must fail loudly rather than vanish from the dashboard."""
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(
            {
                "type": "risk",
                "call_id": "c1",
                "seq": 1,
                "ts": TS,
                "hangup_risk": 0.1,
                "hangup_rsk": 0.9,
            }
        )


@pytest.mark.parametrize("risk", [-0.1, 1.1])
def test_scores_outside_zero_to_one_are_rejected(risk: float) -> None:
    with pytest.raises(ValidationError):
        RiskEvent(call_id="c1", seq=1, ts=TS, hangup_risk=risk)


def test_negative_sequence_numbers_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StageEvent(
            call_id="c1", seq=-1, ts=TS, stage=ScamStage.HOOK, confidence=0.1, pressure_score=0.1
        )

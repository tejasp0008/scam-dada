"""The event schema.

One schema serves three consumers: the dashboard WebSocket payload, the `events` table
(JSONB) and the evidence pack's replay. Keeping them identical means the dashboard can
never drift from what was persisted, and a finished call can be replayed into the live UI
without a second code path.

Events are frozen and carry a per-call monotonic `seq`, so a dashboard that reconnects can
ask for everything after the last sequence number it saw.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import (
    EntitySource,
    EntityType,
    PersonaId,
    ScamStage,
    Speaker,
)
from app.core.state import CallState


class EventType(StrEnum):
    STATE_CHANGE = "state_change"
    TRANSCRIPT = "transcript"
    STAGE = "stage"
    ENTITY = "entity"
    RISK = "risk"
    PERSONA_HANDOFF = "persona_handoff"
    CANARY_MATCH = "canary_match"
    LATENCY = "latency"


class _EventBase(BaseModel):
    """Fields every event carries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    seq: int = Field(ge=0, description="Monotonic within a call; assigned by the emitter.")
    ts: datetime = Field(description="UTC instant the event was emitted.")


class StateChangeEvent(_EventBase):
    type: Literal[EventType.STATE_CHANGE] = EventType.STATE_CHANGE
    from_state: CallState
    to_state: CallState
    reason: str


class TranscriptEvent(_EventBase):
    type: Literal[EventType.TRANSCRIPT] = EventType.TRANSCRIPT
    turn_seq: int = Field(ge=0)
    speaker: Speaker
    text: str
    persona: PersonaId | None = None
    language_code: str | None = Field(
        default=None, description="Prisma's detected_language_code when available."
    )
    audio_duration_ms: int | None = Field(default=None, ge=0)
    # Prisma streaming returns is_final=true on every transcript (it has no partial
    # hypotheses), but the persona's own TTS text is emitted optimistically before the
    # audio finishes, so the flag still earns its place.
    is_final: bool = True


class StageEvent(_EventBase):
    type: Literal[EventType.STAGE] = EventType.STAGE
    stage: ScamStage
    confidence: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)


class EntityEvent(_EventBase):
    type: Literal[EventType.ENTITY] = EventType.ENTITY
    entity_type: EntityType
    value: str
    source: EntitySource
    turn_seq: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RiskEvent(_EventBase):
    type: Literal[EventType.RISK] = EventType.RISK
    hangup_risk: float = Field(ge=0.0, le=1.0)
    signals: tuple[str, ...] = Field(
        default=(), description="Named contributing signals, for an auditable gauge."
    )
    tactic: str | None = Field(
        default=None, description="Retention tactic chosen in response, if any."
    )


class PersonaHandoffEvent(_EventBase):
    type: Literal[EventType.PERSONA_HANDOFF] = EventType.PERSONA_HANDOFF
    from_persona: PersonaId | None
    to_persona: PersonaId
    reason: str


class CanaryMatchEvent(_EventBase):
    type: Literal[EventType.CANARY_MATCH] = EventType.CANARY_MATCH
    identity_id: str
    issued_call_id: str = Field(description="The earlier call this identity was issued to.")
    matched_field: str
    matched_text: str
    score: float = Field(ge=0.0, le=1.0)


class LatencyEvent(_EventBase):
    """Per-turn latency breakdown.

    Each leg is measured from the end of the caller's speech, so the numbers add up to the
    silence the scammer actually heard.
    """

    type: Literal[EventType.LATENCY] = EventType.LATENCY
    turn_seq: int = Field(ge=0)
    stt_final_ms: int | None = Field(default=None, ge=0)
    llm_first_token_ms: int | None = Field(default=None, ge=0)
    tts_first_byte_ms: int | None = Field(default=None, ge=0)
    filler_played: bool = False


DomainEvent = Annotated[
    StateChangeEvent
    | TranscriptEvent
    | StageEvent
    | EntityEvent
    | RiskEvent
    | PersonaHandoffEvent
    | CanaryMatchEvent
    | LatencyEvent,
    Field(discriminator="type"),
]
"""Discriminated union used for the WebSocket payload and for replay from JSONB."""

"""Domain vocabulary for a baited call.

Pure data. Nothing here may import FastAPI, SQLAlchemy, Redis or a provider SDK: the
whole point of `app.core` is that it can be unit-tested without a running process.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Speaker(StrEnum):
    """Who produced a turn."""

    CALLER = "caller"
    PERSONA = "persona"
    SYSTEM = "system"


class PersonaId(StrEnum):
    """The family relay, in handoff order."""

    DADI = "dadi"
    DADA = "dada"
    GRANDSON = "grandson"


class ScamStage(StrEnum):
    """Stages of a social-engineering script, in the order they usually appear."""

    HOOK = "hook"
    AUTHORITY = "authority"
    FEAR = "fear"
    ISOLATION = "isolation"
    PAYMENT = "payment"


class EntityType(StrEnum):
    """Intelligence worth extracting from a scammer's script."""

    UPI_ID = "upi_id"
    PHONE = "phone"
    IFSC = "ifsc"
    AMOUNT = "amount"
    OFFICER_NAME = "officer_name"
    ORGANISATION = "organisation"
    CASE_NUMBER = "case_number"


class EntitySource(StrEnum):
    """How an entity was found. Regex hits are deterministic; LLM hits are not."""

    REGEX = "regex"
    LLM = "llm"


class Turn(BaseModel):
    """One utterance by one party."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0, description="Monotonic turn index within the call.")
    speaker: Speaker
    text: str
    persona: PersonaId | None = Field(default=None, description="Set only when speaker is PERSONA.")
    language_code: str | None = None
    started_at_ms: int = Field(ge=0, description="Offset from call start.")
    duration_ms: int | None = Field(default=None, ge=0)


class Entity(BaseModel):
    """A single extracted identifier or claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_type: EntityType
    value: str
    source: EntitySource
    turn_seq: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CanaryIdentity(BaseModel):
    """A deliberately fictional victim identity issued to exactly one call.

    Every field must be visibly fake. If a later call's transcript mentions one of these,
    the two calls share a lead list, which is the "identity leak" signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str
    full_name: str
    upi_handle: str
    locality: str
    phone: str
    issued_at_ms: int = Field(ge=0)

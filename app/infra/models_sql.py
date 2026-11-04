"""SQLAlchemy 2.0 tables.

Separate from `app.core.models` on purpose: the domain models are frozen Pydantic types
with no persistence concerns, and these are rows. The event payload column is JSONB
holding a serialised `DomainEvent`, so adding an event type needs no migration.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: JSONB on Postgres, plain JSON elsewhere, so the schema is also usable under SQLite in tests.
JsonB = JSONB().with_variant(JSON(), "sqlite")


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    final_state: Mapped[str | None] = mapped_column(String(32))
    ended_reason: Mapped[str | None] = mapped_column(String(64))
    # Ordered persona ids, so the family relay is visible without replaying events.
    persona_sequence: Mapped[list[str] | None] = mapped_column(JsonB)
    canary_identity_id: Mapped[str | None] = mapped_column(String(36))
    scammer_seconds_wasted: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="browser")

    turns: Mapped[list[Turn]] = relationship(back_populates="call", cascade="all, delete-orphan")
    events: Mapped[list[Event]] = relationship(back_populates="call", cascade="all, delete-orphan")


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (UniqueConstraint("call_id", "seq", name="uq_turns_call_seq"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(String(16), nullable=False)
    persona: Mapped[str | None] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language_code: Mapped[str | None] = mapped_column(String(16))
    started_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    # Latency legs live on the turn so the dashboard panel survives a restart.
    stt_final_ms: Mapped[int | None] = mapped_column(Integer)
    llm_first_token_ms: Mapped[int | None] = mapped_column(Integer)
    tts_first_byte_ms: Mapped[int | None] = mapped_column(Integer)

    call: Mapped[Call] = relationship(back_populates="turns")
    entities: Mapped[list[Entity]] = relationship(back_populates="turn")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("call_id", "seq", name="uq_events_call_seq"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JsonB, nullable=False)

    call: Mapped[Call] = relationship(back_populates="events")


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_id: Mapped[str | None] = mapped_column(ForeignKey("turns.id", ondelete="SET NULL"))
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    turn: Mapped[Turn | None] = relationship(back_populates="entities")


class CanaryIdentity(Base):
    __tablename__ = "canary_identities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    issued_call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    identity: Mapped[dict[str, object]] = mapped_column(JsonB, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CanaryMatch(Base):
    __tablename__ = "canary_matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    identity_id: Mapped[str] = mapped_column(
        ForeignKey("canary_identities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    matched_call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    matched_field: Mapped[str] = mapped_column(String(32), nullable=False)
    matched_text: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ApiUsage(Base):
    """One row per outbound provider call, including cache hits at zero units.

    Cache hits are recorded rather than skipped so "credits saved by caching" is a measured
    number instead of a claim.
    """

    __tablename__ = "api_usage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    call_id: Mapped[str | None] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL"), index=True
    )
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    unit_type: Mapped[str] = mapped_column(String(16), nullable=False)
    units: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

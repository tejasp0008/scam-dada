"""Baseline schema: calls, turns, events, entities, canaries, usage.

One migration for the whole schema. The contest week does not have room for incremental
migrations, and a single reviewable baseline is easier to reason about than a chain of
seven.

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "calls",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("final_state", sa.String(length=32), nullable=True),
        sa.Column("ended_reason", sa.String(length=64), nullable=True),
        sa.Column("persona_sequence", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("canary_identity_id", sa.String(length=36), nullable=True),
        sa.Column("scammer_seconds_wasted", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "turns",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(length=16), nullable=False),
        sa.Column("persona", sa.String(length=16), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language_code", sa.String(length=16), nullable=True),
        sa.Column("started_at_ms", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("stt_final_ms", sa.Integer(), nullable=True),
        sa.Column("llm_first_token_ms", sa.Integer(), nullable=True),
        sa.Column("tts_first_byte_ms", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "seq", name="uq_turns_call_seq"),
    )
    op.create_index("ix_turns_call_id", "turns", ["call_id"])

    op.create_table(
        "events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "seq", name="uq_events_call_seq"),
    )
    op.create_index("ix_events_call_id", "events", ["call_id"])
    op.create_index("ix_events_type", "events", ["type"])

    op.create_table(
        "entities",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("turn_id", sa.String(length=36), nullable=True),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=8), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["turns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_entities_call_id", "entities", ["call_id"])

    op.create_table(
        "canary_identities",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("issued_call_id", sa.String(length=36), nullable=False),
        sa.Column("identity", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["issued_call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_canary_identities_issued_call_id", "canary_identities", ["issued_call_id"])

    op.create_table(
        "canary_matches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("identity_id", sa.String(length=36), nullable=False),
        sa.Column("matched_call_id", sa.String(length=36), nullable=False),
        sa.Column("matched_field", sa.String(length=32), nullable=False),
        sa.Column("matched_text", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["identity_id"], ["canary_identities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["matched_call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_canary_matches_identity_id", "canary_matches", ["identity_id"])
    op.create_index("ix_canary_matches_matched_call_id", "canary_matches", ["matched_call_id"])

    op.create_table(
        "api_usage",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=True),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("unit_type", sa.String(length=16), nullable=False),
        sa.Column("units", sa.Integer(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_usage_call_id", "api_usage", ["call_id"])
    op.create_index("ix_api_usage_provider", "api_usage", ["provider"])


def downgrade() -> None:
    op.drop_table("api_usage")
    op.drop_table("canary_matches")
    op.drop_table("canary_identities")
    op.drop_table("entities")
    op.drop_table("events")
    op.drop_table("turns")
    op.drop_table("calls")

"""Record provider usage in credits as well as raw units.

Gnani's usage dashboard bills in credits (₹1 each) and bills STT in rounded-up minutes
rather than seconds, so the original `unit_type`/`units` pair could not represent a charge:
2.7 seconds of audio is billed as 0.05 minutes, and an integer column cannot hold either
the fraction or the distinction.

The table is replaced rather than migrated column by column. At this point it holds only
rows from credential smoke tests, and preserving six throwaway rows is not worth a
multi-step column dance.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("api_usage")
    op.create_table(
        "api_usage",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=True),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("pool", sa.String(length=16), nullable=False),
        sa.Column("raw_unit", sa.String(length=16), nullable=False),
        sa.Column("raw_units", sa.Float(), nullable=False),
        sa.Column("billed_unit", sa.String(length=16), nullable=False),
        sa.Column("billed_units", sa.Float(), nullable=False),
        sa.Column("credits", sa.Float(), nullable=False),
        sa.Column("credits_estimated", sa.Boolean(), nullable=False),
        sa.Column("avoided_credits", sa.Float(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_usage_call_id", "api_usage", ["call_id"])
    op.create_index("ix_api_usage_provider", "api_usage", ["provider"])
    op.create_index("ix_api_usage_pool", "api_usage", ["pool"])


def downgrade() -> None:
    op.drop_table("api_usage")
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

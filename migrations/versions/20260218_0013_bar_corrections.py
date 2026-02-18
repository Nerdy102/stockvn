"""add bar corrections and bars intraday uniqueness on start/end

Revision ID: 20260218_0013
Revises: 20260218_0012
Create Date: 2026-02-18 00:13:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0013"
down_revision = "20260218_0012"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "bars_intraday"):
        if _has_index(inspector, "bars_intraday", "ux_bars_intraday_symbol_tf_start"):
            op.drop_index("ux_bars_intraday_symbol_tf_start", table_name="bars_intraday")
        if not _has_index(inspector, "bars_intraday", "ux_bars_intraday_symbol_tf_start_end"):
            op.create_index(
                "ux_bars_intraday_symbol_tf_start_end",
                "bars_intraday",
                ["symbol", "timeframe", "start_ts", "end_ts"],
                unique=True,
            )

    if not _has_table(inspector, "bar_corrections"):
        op.create_table(
            "bar_corrections",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("bar_start_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("bar_end_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("reason", sa.String(), nullable=False),
            sa.Column("original_event_id", sa.String(), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "bar_corrections") and not _has_index(
        inspector, "bar_corrections", "ix_bar_corrections_symbol_tf_start"
    ):
        op.create_index(
            "ix_bar_corrections_symbol_tf_start",
            "bar_corrections",
            ["symbol", "timeframe", "bar_start_ts"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "bar_corrections"):
        if _has_index(inspector, "bar_corrections", "ix_bar_corrections_symbol_tf_start"):
            op.drop_index("ix_bar_corrections_symbol_tf_start", table_name="bar_corrections")
        op.drop_table("bar_corrections")

    if _has_table(inspector, "bars_intraday"):
        if _has_index(inspector, "bars_intraday", "ux_bars_intraday_symbol_tf_start_end"):
            op.drop_index("ux_bars_intraday_symbol_tf_start_end", table_name="bars_intraday")
        if not _has_index(inspector, "bars_intraday", "ux_bars_intraday_symbol_tf_start"):
            op.create_index(
                "ux_bars_intraday_symbol_tf_start",
                "bars_intraday",
                ["symbol", "timeframe", "start_ts"],
                unique=True,
            )

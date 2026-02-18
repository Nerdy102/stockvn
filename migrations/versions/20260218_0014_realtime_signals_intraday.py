"""add realtime signals intraday tables

Revision ID: 20260218_0014
Revises: 20260218_0013
Create Date: 2026-02-18 00:14:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0014"
down_revision = "20260218_0013"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "signals_intraday"):
        op.create_table(
            "signals_intraday",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("end_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("indicators_json", sa.JSON(), nullable=False),
            sa.Column("setups_json", sa.JSON(), nullable=False),
            sa.Column("signal_hash", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "signals_intraday"):
        if not _has_index(inspector, "signals_intraday", "ix_signals_intraday_symbol_tf_end"):
            op.create_index(
                "ix_signals_intraday_symbol_tf_end",
                "signals_intraday",
                ["symbol", "timeframe", "end_ts"],
                unique=False,
            )
        if not _has_index(inspector, "signals_intraday", "ux_signals_intraday_symbol_tf_end"):
            op.create_index(
                "ux_signals_intraday_symbol_tf_end",
                "signals_intraday",
                ["symbol", "timeframe", "end_ts"],
                unique=True,
            )

    if not _has_table(inspector, "alerts_intraday"):
        op.create_table(
            "alerts_intraday",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("end_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("alert_key", sa.String(), nullable=False),
            sa.Column("severity", sa.Integer(), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "alerts_intraday"):
        if not _has_index(inspector, "alerts_intraday", "ix_alerts_intraday_symbol_tf_end"):
            op.create_index(
                "ix_alerts_intraday_symbol_tf_end",
                "alerts_intraday",
                ["symbol", "timeframe", "end_ts"],
                unique=False,
            )
        if not _has_index(inspector, "alerts_intraday", "ux_alerts_intraday_symbol_tf_end_key"):
            op.create_index(
                "ux_alerts_intraday_symbol_tf_end_key",
                "alerts_intraday",
                ["symbol", "timeframe", "end_ts", "alert_key"],
                unique=True,
            )

    if not _has_table(inspector, "alert_cooldowns"):
        op.create_table(
            "alert_cooldowns",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("rule_key", sa.String(), nullable=False),
            sa.Column("last_bar_index", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "alert_cooldowns") and not _has_index(
        inspector, "alert_cooldowns", "ux_alert_cooldowns_symbol_tf_rule"
    ):
        op.create_index(
            "ux_alert_cooldowns_symbol_tf_rule",
            "alert_cooldowns",
            ["symbol", "timeframe", "rule_key"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table, indexes in [
        ("alert_cooldowns", ["ux_alert_cooldowns_symbol_tf_rule"]),
        (
            "alerts_intraday",
            ["ux_alerts_intraday_symbol_tf_end_key", "ix_alerts_intraday_symbol_tf_end"],
        ),
        (
            "signals_intraday",
            ["ux_signals_intraday_symbol_tf_end", "ix_signals_intraday_symbol_tf_end"],
        ),
    ]:
        if _has_table(inspector, table):
            for idx in indexes:
                if _has_index(inspector, table, idx):
                    op.drop_index(idx, table_name=table)
            op.drop_table(table)
            inspector = sa.inspect(bind)

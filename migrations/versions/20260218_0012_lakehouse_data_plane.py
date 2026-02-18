"""add lakehouse data plane tables

Revision ID: 20260218_0012
Revises: 20260218_0011
Create Date: 2026-02-18 00:12:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0012"
down_revision = "20260218_0011"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "schema_versions"):
        op.create_table(
            "schema_versions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("channel", sa.String(), nullable=False),
            sa.Column("version_hash", sa.String(), nullable=False),
            sa.Column("keys_json", sa.JSON(), nullable=False),
            sa.Column("first_seen_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "schema_versions"):
        if not _has_index(inspector, "schema_versions", "ix_schema_versions_source_channel_seen"):
            op.create_index(
                "ix_schema_versions_source_channel_seen",
                "schema_versions",
                ["source", "channel", "first_seen_at"],
                unique=False,
            )
        if not _has_index(inspector, "schema_versions", "ux_schema_versions_scope_hash"):
            op.create_index(
                "ux_schema_versions_scope_hash",
                "schema_versions",
                ["source", "channel", "version_hash"],
                unique=True,
            )

    if not _has_table(inspector, "dq_metrics"):
        op.create_table(
            "dq_metrics",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dt", sa.Date(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("channel", sa.String(), nullable=False),
            sa.Column("missing_rate", sa.Float(), nullable=False),
            sa.Column("duplicate_rate", sa.Float(), nullable=False),
            sa.Column("ohlc_invariant_rate", sa.Float(), nullable=False),
            sa.Column("psi", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "dq_metrics") and not _has_index(
        inspector, "dq_metrics", "ix_dq_metrics_dt_source_channel"
    ):
        op.create_index(
            "ix_dq_metrics_dt_source_channel",
            "dq_metrics",
            ["dt", "source", "channel"],
            unique=False,
        )

    if not _has_table(inspector, "canonical_trades"):
        op.create_table(
            "canonical_trades",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("exchange", sa.String(), nullable=False),
            sa.Column("instrument", sa.String(), nullable=False),
            sa.Column("ts_utc", sa.DateTime(timezone=False), nullable=False),
            sa.Column("price", sa.Float(), nullable=False),
            sa.Column("qty", sa.Float(), nullable=False),
            sa.Column("payload_hash", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "canonical_trades"):
        if not _has_index(inspector, "canonical_trades", "ix_canonical_trades_symbol_ts"):
            op.create_index(
                "ix_canonical_trades_symbol_ts",
                "canonical_trades",
                ["symbol", "ts_utc"],
                unique=False,
            )
        if not _has_index(inspector, "canonical_trades", "ux_canonical_trades_event"):
            op.create_index(
                "ux_canonical_trades_event", "canonical_trades", ["event_id"], unique=True
            )

    if not _has_table(inspector, "canonical_quotes"):
        op.create_table(
            "canonical_quotes",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("ts_utc", sa.DateTime(timezone=False), nullable=False),
            sa.Column("bid_px", sa.Float(), nullable=False),
            sa.Column("bid_qty", sa.Float(), nullable=False),
            sa.Column("ask_px", sa.Float(), nullable=False),
            sa.Column("ask_qty", sa.Float(), nullable=False),
            sa.Column("payload_hash", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "canonical_quotes"):
        if not _has_index(inspector, "canonical_quotes", "ix_canonical_quotes_symbol_ts"):
            op.create_index(
                "ix_canonical_quotes_symbol_ts",
                "canonical_quotes",
                ["symbol", "ts_utc"],
                unique=False,
            )
        if not _has_index(inspector, "canonical_quotes", "ux_canonical_quotes_event"):
            op.create_index(
                "ux_canonical_quotes_event", "canonical_quotes", ["event_id"], unique=True
            )

    if not _has_table(inspector, "bars_intraday"):
        op.create_table(
            "bars_intraday",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("start_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("end_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("o", sa.Float(), nullable=False),
            sa.Column("h", sa.Float(), nullable=False),
            sa.Column("l", sa.Float(), nullable=False),
            sa.Column("c", sa.Float(), nullable=False),
            sa.Column("v", sa.Float(), nullable=False),
            sa.Column("n_trades", sa.Integer(), nullable=False),
            sa.Column("vwap", sa.Float(), nullable=False),
            sa.Column("finalized", sa.Boolean(), nullable=False),
            sa.Column("build_ts", sa.DateTime(), nullable=False),
            sa.Column("bar_hash", sa.String(), nullable=False),
            sa.Column("lineage_payload_hashes_json", sa.JSON(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "bars_intraday"):
        if not _has_index(inspector, "bars_intraday", "ix_bars_intraday_symbol_tf_start"):
            op.create_index(
                "ix_bars_intraday_symbol_tf_start",
                "bars_intraday",
                ["symbol", "timeframe", "start_ts"],
                unique=False,
            )
        if not _has_index(inspector, "bars_intraday", "ux_bars_intraday_symbol_tf_start"):
            op.create_index(
                "ux_bars_intraday_symbol_tf_start",
                "bars_intraday",
                ["symbol", "timeframe", "start_ts"],
                unique=True,
            )

    if not _has_table(inspector, "feature_snapshots"):
        op.create_table(
            "feature_snapshots",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("as_of_ts", sa.DateTime(timezone=False), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("features_json", sa.JSON(), nullable=False),
            sa.Column("feature_hash", sa.String(), nullable=False),
            sa.Column("lineage_json", sa.JSON(), nullable=False),
            sa.Column("as_of_date", sa.Date(), nullable=False),
            sa.Column("matured_date", sa.Date(), nullable=False),
            sa.Column("public_date", sa.Date(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "feature_snapshots"):
        if not _has_index(inspector, "feature_snapshots", "ix_feature_snapshots_symbol_tf_asof"):
            op.create_index(
                "ix_feature_snapshots_symbol_tf_asof",
                "feature_snapshots",
                ["symbol", "timeframe", "as_of_ts"],
                unique=False,
            )
        if not _has_index(inspector, "feature_snapshots", "ux_feature_snapshots_symbol_tf_asof"):
            op.create_index(
                "ux_feature_snapshots_symbol_tf_asof",
                "feature_snapshots",
                ["symbol", "timeframe", "as_of_ts"],
                unique=True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table, indexes in [
        (
            "feature_snapshots",
            ["ux_feature_snapshots_symbol_tf_asof", "ix_feature_snapshots_symbol_tf_asof"],
        ),
        ("bars_intraday", ["ux_bars_intraday_symbol_tf_start", "ix_bars_intraday_symbol_tf_start"]),
        ("canonical_quotes", ["ux_canonical_quotes_event", "ix_canonical_quotes_symbol_ts"]),
        ("canonical_trades", ["ux_canonical_trades_event", "ix_canonical_trades_symbol_ts"]),
        ("dq_metrics", ["ix_dq_metrics_dt_source_channel"]),
        (
            "schema_versions",
            ["ux_schema_versions_scope_hash", "ix_schema_versions_source_channel_seen"],
        ),
    ]:
        if _has_table(inspector, table):
            for idx in indexes:
                if _has_index(inspector, table, idx):
                    op.drop_index(idx, table_name=table)
            op.drop_table(table)
            inspector = sa.inspect(bind)

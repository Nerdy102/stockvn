"""add daily feature tables and flow room columns

Revision ID: 20260218_0002
Revises: 20260218_0001
Create Date: 2026-02-18 10:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0002"
down_revision = "20260218_0001"
branch_labels = None
depends_on = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "market_daily_meta") and not _has_column(
        inspector, "market_daily_meta", "current_room"
    ):
        with op.batch_alter_table("market_daily_meta") as batch_op:
            batch_op.add_column(sa.Column("current_room", sa.Float(), nullable=True))

    if _has_table(inspector, "market_daily_meta") and not _has_column(
        inspector, "market_daily_meta", "total_room"
    ):
        with op.batch_alter_table("market_daily_meta") as batch_op:
            batch_op.add_column(sa.Column("total_room", sa.Float(), nullable=True))

    if not _has_table(inspector, "daily_flow_features"):
        op.create_table(
            "daily_flow_features",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("source", sa.String(), nullable=False, server_default="derived"),
            sa.Column("net_foreign_val_day", sa.Float(), nullable=False, server_default="0"),
            sa.Column("net_foreign_val_5d", sa.Float(), nullable=False, server_default="0"),
            sa.Column("net_foreign_val_20d", sa.Float(), nullable=False, server_default="0"),
            sa.Column("foreign_flow_intensity", sa.Float(), nullable=False, server_default="0"),
            sa.Column("foreign_room_util", sa.Float(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_flow_features") and not _has_index(
        inspector, "daily_flow_features", "ix_daily_flow_symbol_date"
    ):
        op.create_index("ix_daily_flow_symbol_date", "daily_flow_features", ["symbol", "date"])
    if _has_table(inspector, "daily_flow_features") and not _has_index(
        inspector, "daily_flow_features", "ux_daily_flow_symbol_date_source"
    ):
        op.create_index(
            "ux_daily_flow_symbol_date_source",
            "daily_flow_features",
            ["symbol", "date", "source"],
            unique=True,
        )

    if not _has_table(inspector, "daily_orderbook_features"):
        op.create_table(
            "daily_orderbook_features",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("source", sa.String(), nullable=False, server_default="derived"),
            sa.Column("imb_1_day", sa.Float(), nullable=False, server_default="0"),
            sa.Column("imb_3_day", sa.Float(), nullable=False, server_default="0"),
            sa.Column("spread_day", sa.Float(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_orderbook_features") and not _has_index(
        inspector, "daily_orderbook_features", "ix_daily_orderbook_symbol_date"
    ):
        op.create_index(
            "ix_daily_orderbook_symbol_date", "daily_orderbook_features", ["symbol", "date"]
        )
    if _has_table(inspector, "daily_orderbook_features") and not _has_index(
        inspector, "daily_orderbook_features", "ux_daily_orderbook_symbol_date_source"
    ):
        op.create_index(
            "ux_daily_orderbook_symbol_date_source",
            "daily_orderbook_features",
            ["symbol", "date", "source"],
            unique=True,
        )

    if not _has_table(inspector, "daily_intraday_features"):
        op.create_table(
            "daily_intraday_features",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("source", sa.String(), nullable=False, server_default="derived"),
            sa.Column("rv_day", sa.Float(), nullable=False, server_default="0"),
            sa.Column("vol_first_hour_ratio", sa.Float(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_intraday_features") and not _has_index(
        inspector, "daily_intraday_features", "ix_daily_intraday_symbol_date"
    ):
        op.create_index("ix_daily_intraday_symbol_date", "daily_intraday_features", ["symbol", "date"])
    if _has_table(inspector, "daily_intraday_features") and not _has_index(
        inspector, "daily_intraday_features", "ux_daily_intraday_symbol_date_source"
    ):
        op.create_index(
            "ux_daily_intraday_symbol_date_source",
            "daily_intraday_features",
            ["symbol", "date", "source"],
            unique=True,
        )

    if not _has_table(inspector, "feature_last_processed"):
        op.create_table(
            "feature_last_processed",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("feature_name", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False, server_default=""),
            sa.Column("last_date", sa.Date(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "feature_last_processed") and not _has_index(
        inspector, "feature_last_processed", "ix_feature_last_processed_feature_name"
    ):
        op.create_index(
            "ix_feature_last_processed_feature_name", "feature_last_processed", ["feature_name"]
        )
    if _has_table(inspector, "feature_last_processed") and not _has_index(
        inspector, "feature_last_processed", "ix_feature_last_processed_symbol"
    ):
        op.create_index("ix_feature_last_processed_symbol", "feature_last_processed", ["symbol"])
    if _has_table(inspector, "feature_last_processed") and not _has_index(
        inspector, "feature_last_processed", "ux_feature_last_processed"
    ):
        op.create_index(
            "ux_feature_last_processed",
            "feature_last_processed",
            ["feature_name", "symbol"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "feature_last_processed"):
        if _has_index(inspector, "feature_last_processed", "ux_feature_last_processed"):
            op.drop_index("ux_feature_last_processed", table_name="feature_last_processed")
        if _has_index(inspector, "feature_last_processed", "ix_feature_last_processed_symbol"):
            op.drop_index("ix_feature_last_processed_symbol", table_name="feature_last_processed")
        if _has_index(inspector, "feature_last_processed", "ix_feature_last_processed_feature_name"):
            op.drop_index("ix_feature_last_processed_feature_name", table_name="feature_last_processed")
        op.drop_table("feature_last_processed")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_intraday_features"):
        if _has_index(inspector, "daily_intraday_features", "ux_daily_intraday_symbol_date_source"):
            op.drop_index("ux_daily_intraday_symbol_date_source", table_name="daily_intraday_features")
        if _has_index(inspector, "daily_intraday_features", "ix_daily_intraday_symbol_date"):
            op.drop_index("ix_daily_intraday_symbol_date", table_name="daily_intraday_features")
        op.drop_table("daily_intraday_features")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_orderbook_features"):
        if _has_index(inspector, "daily_orderbook_features", "ux_daily_orderbook_symbol_date_source"):
            op.drop_index("ux_daily_orderbook_symbol_date_source", table_name="daily_orderbook_features")
        if _has_index(inspector, "daily_orderbook_features", "ix_daily_orderbook_symbol_date"):
            op.drop_index("ix_daily_orderbook_symbol_date", table_name="daily_orderbook_features")
        op.drop_table("daily_orderbook_features")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "daily_flow_features"):
        if _has_index(inspector, "daily_flow_features", "ux_daily_flow_symbol_date_source"):
            op.drop_index("ux_daily_flow_symbol_date_source", table_name="daily_flow_features")
        if _has_index(inspector, "daily_flow_features", "ix_daily_flow_symbol_date"):
            op.drop_index("ix_daily_flow_symbol_date", table_name="daily_flow_features")
        op.drop_table("daily_flow_features")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "market_daily_meta"):
        cols = {c["name"] for c in inspector.get_columns("market_daily_meta")}
        with op.batch_alter_table("market_daily_meta") as batch_op:
            if "total_room" in cols:
                batch_op.drop_column("total_room")
            if "current_room" in cols:
                batch_op.drop_column("current_room")

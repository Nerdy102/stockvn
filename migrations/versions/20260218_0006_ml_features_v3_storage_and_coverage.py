"""ml_features v3 columnar storage and feature coverage table

Revision ID: 20260218_0006
Revises: 20260218_0005
Create Date: 2026-02-18 00:06:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260218_0006"
down_revision = "20260218_0005"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    cols = inspector.get_columns(table_name)
    return any(c["name"] == column_name for c in cols)


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def _ensure_column(inspector: sa.Inspector, table: str, column: sa.Column) -> None:
    if _has_table(inspector, table) and not _has_column(inspector, table, str(column.name)):
        op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    numeric_cols = [
        "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "rev_5d",
        "vol_20d", "vol_60d", "vol_120d", "atr14_pct", "adv20_value", "adv20_vol",
        "spread_proxy", "limit_hit_20d", "rsi14", "macd_hist", "ema20_gt_ema50", "close_gt_ema50",
        "ema50_slope", "value_score_z", "quality_score_z", "momentum_score_z", "lowvol_score_z",
        "dividend_score_z", "regime_trend_up", "regime_sideways", "regime_risk_off",
        "net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util",
        "imb_1_day", "imb_3_day", "spread_day", "rv_day", "vol_first_hour_ratio",
        "fundamental_public_date_is_assumed", "fundamental_public_date_limitation_flag", "y_excess", "y_rank_z",
    ]

    if _has_table(inspector, "ml_features"):
        _ensure_column(inspector, "ml_features", sa.Column("created_at", sa.DateTime(), nullable=True))
        _ensure_column(inspector, "ml_features", sa.Column("data_coverage_score", sa.Float(), nullable=False, server_default="0"))
        for c in numeric_cols:
            _ensure_column(inspector, "ml_features", sa.Column(c, sa.Float(), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_table(inspector, "feature_coverage"):
        op.create_table(
            "feature_coverage",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("feature_version", sa.String(), nullable=False, server_default="v3"),
            sa.Column("metrics_json", sa.JSON(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "feature_coverage") and not _has_index(inspector, "feature_coverage", "ux_feature_coverage"):
        op.create_index("ux_feature_coverage", "feature_coverage", ["date", "feature_version"], unique=True)

    if _has_table(inspector, "feature_coverage") and not _has_index(inspector, "feature_coverage", "ix_feature_coverage_date"):
        op.create_index("ix_feature_coverage_date", "feature_coverage", ["date"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "feature_coverage"):
        if _has_index(inspector, "feature_coverage", "ix_feature_coverage_date"):
            op.drop_index("ix_feature_coverage_date", table_name="feature_coverage")
        if _has_index(inspector, "feature_coverage", "ux_feature_coverage"):
            op.drop_index("ux_feature_coverage", table_name="feature_coverage")
        op.drop_table("feature_coverage")

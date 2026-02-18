"""add conformal tables for alpha_v3_cp

Revision ID: 20260218_0007
Revises: 20260218_0006
Create Date: 2026-02-18 00:07:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260218_0007"
down_revision = "20260218_0006"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "conformal_state"):
        op.create_table(
            "conformal_state",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("bucket_id", sa.Integer(), nullable=False),
            sa.Column("alpha_b", sa.Float(), nullable=False, server_default="0.2"),
            sa.Column("miss_ema", sa.Float(), nullable=False, server_default="0.2"),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "conformal_state") and not _has_index(inspector, "conformal_state", "ux_conformal_state"):
        op.create_index("ux_conformal_state", "conformal_state", ["model_id", "bucket_id"], unique=True)

    if not _has_table(inspector, "conformal_residuals"):
        op.create_table(
            "conformal_residuals",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("bucket_id", sa.Integer(), nullable=False),
            sa.Column("abs_residual", sa.Float(), nullable=False),
            sa.Column("miss", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "conformal_residuals") and not _has_index(inspector, "conformal_residuals", "ux_conformal_residual"):
        op.create_index("ux_conformal_residual", "conformal_residuals", ["model_id", "date", "symbol"], unique=True)

    if not _has_table(inspector, "conformal_bucket_spec"):
        op.create_table(
            "conformal_bucket_spec",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("month_start", sa.Date(), nullable=False),
            sa.Column("bucket_id", sa.Integer(), nullable=False),
            sa.Column("low", sa.Float(), nullable=True),
            sa.Column("high", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "conformal_bucket_spec") and not _has_index(inspector, "conformal_bucket_spec", "ux_conformal_bucket_spec"):
        op.create_index("ux_conformal_bucket_spec", "conformal_bucket_spec", ["model_id", "month_start", "bucket_id"], unique=True)

    if not _has_table(inspector, "conformal_coverage_daily"):
        op.create_table(
            "conformal_coverage_daily",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("bucket_id", sa.Integer(), nullable=False),
            sa.Column("coverage", sa.Float(), nullable=False),
            sa.Column("interval_half_width", sa.Float(), nullable=False),
            sa.Column("count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "conformal_coverage_daily") and not _has_index(inspector, "conformal_coverage_daily", "ux_conformal_coverage_daily"):
        op.create_index("ux_conformal_coverage_daily", "conformal_coverage_daily", ["model_id", "date", "bucket_id"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "conformal_coverage_daily"):
        if _has_index(inspector, "conformal_coverage_daily", "ux_conformal_coverage_daily"):
            op.drop_index("ux_conformal_coverage_daily", table_name="conformal_coverage_daily")
        op.drop_table("conformal_coverage_daily")

    if _has_table(inspector, "conformal_bucket_spec"):
        if _has_index(inspector, "conformal_bucket_spec", "ux_conformal_bucket_spec"):
            op.drop_index("ux_conformal_bucket_spec", table_name="conformal_bucket_spec")
        op.drop_table("conformal_bucket_spec")

    if _has_table(inspector, "conformal_residuals"):
        if _has_index(inspector, "conformal_residuals", "ux_conformal_residual"):
            op.drop_index("ux_conformal_residual", table_name="conformal_residuals")
        op.drop_table("conformal_residuals")

    if _has_table(inspector, "conformal_state"):
        if _has_index(inspector, "conformal_state", "ux_conformal_state"):
            op.drop_index("ux_conformal_state", table_name="conformal_state")
        op.drop_table("conformal_state")

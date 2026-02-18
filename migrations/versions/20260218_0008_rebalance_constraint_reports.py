"""add rebalance constraint reports

Revision ID: 20260218_0008
Revises: 20260218_0007
Create Date: 2026-02-18 00:08:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260218_0008"
down_revision = "20260218_0007"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "rebalance_constraint_reports"):
        op.create_table(
            "rebalance_constraint_reports",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("as_of_date", sa.Date(), nullable=False),
            sa.Column("run_tag", sa.String(), nullable=False, server_default="alpha_v3"),
            sa.Column("report_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "rebalance_constraint_reports") and not _has_index(
        inspector, "rebalance_constraint_reports", "ix_rebalance_constraint_reports_date_tag"
    ):
        op.create_index(
            "ix_rebalance_constraint_reports_date_tag",
            "rebalance_constraint_reports",
            ["as_of_date", "run_tag"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "rebalance_constraint_reports"):
        if _has_index(inspector, "rebalance_constraint_reports", "ix_rebalance_constraint_reports_date_tag"):
            op.drop_index("ix_rebalance_constraint_reports_date_tag", table_name="rebalance_constraint_reports")
        op.drop_table("rebalance_constraint_reports")

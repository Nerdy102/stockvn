"""add ml_labels table for alpha v3

Revision ID: 20260218_0003
Revises: 20260218_0002
Create Date: 2026-02-18 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0003"
down_revision = "20260218_0002"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "ml_labels"):
        op.create_table(
            "ml_labels",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("y_excess", sa.Float(), nullable=False),
            sa.Column("y_rank_z", sa.Float(), nullable=False),
            sa.Column("label_version", sa.String(), nullable=False, server_default="v3"),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "ml_labels") and not _has_index(
        inspector, "ml_labels", "ix_ml_labels_symbol_date"
    ):
        op.create_index("ix_ml_labels_symbol_date", "ml_labels", ["symbol", "date"])

    if _has_table(inspector, "ml_labels") and not _has_index(inspector, "ml_labels", "ix_ml_labels_symbol"):
        op.create_index("ix_ml_labels_symbol", "ml_labels", ["symbol"])

    if _has_table(inspector, "ml_labels") and not _has_index(inspector, "ml_labels", "ix_ml_labels_date"):
        op.create_index("ix_ml_labels_date", "ml_labels", ["date"])

    if _has_table(inspector, "ml_labels") and not _has_index(inspector, "ml_labels", "ix_ml_labels_label_version"):
        op.create_index("ix_ml_labels_label_version", "ml_labels", ["label_version"])

    if _has_table(inspector, "ml_labels") and not _has_index(inspector, "ml_labels", "ux_ml_labels"):
        op.create_index(
            "ux_ml_labels",
            "ml_labels",
            ["symbol", "date", "label_version"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ml_labels"):
        if _has_index(inspector, "ml_labels", "ux_ml_labels"):
            op.drop_index("ux_ml_labels", table_name="ml_labels")
        if _has_index(inspector, "ml_labels", "ix_ml_labels_label_version"):
            op.drop_index("ix_ml_labels_label_version", table_name="ml_labels")
        if _has_index(inspector, "ml_labels", "ix_ml_labels_date"):
            op.drop_index("ix_ml_labels_date", table_name="ml_labels")
        if _has_index(inspector, "ml_labels", "ix_ml_labels_symbol"):
            op.drop_index("ix_ml_labels_symbol", table_name="ml_labels")
        if _has_index(inspector, "ml_labels", "ix_ml_labels_symbol_date"):
            op.drop_index("ix_ml_labels_symbol_date", table_name="ml_labels")
        op.drop_table("ml_labels")

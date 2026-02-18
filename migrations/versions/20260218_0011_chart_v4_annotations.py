"""add chart v4 annotations tables

Revision ID: 20260218_0011
Revises: 20260218_0010
Create Date: 2026-02-18 00:11:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0011"
down_revision = "20260218_0010"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "user_annotations_v2"):
        op.create_table(
            "user_annotations_v2",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("workspace_id", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("timeframe", sa.String(), nullable=False),
            sa.Column("start_date", sa.Date(), nullable=False),
            sa.Column("end_date", sa.Date(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("shapes_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "user_annotations_v2") and not _has_index(
        inspector, "user_annotations_v2", "ix_user_annotations_lookup"
    ):
        op.create_index(
            "ix_user_annotations_lookup",
            "user_annotations_v2",
            ["workspace_id", "symbol", "timeframe", "version"],
            unique=False,
        )
    if _has_table(inspector, "user_annotations_v2") and not _has_index(
        inspector, "user_annotations_v2", "ix_user_annotations_window"
    ):
        op.create_index(
            "ix_user_annotations_window",
            "user_annotations_v2",
            ["start_date", "end_date"],
            unique=False,
        )

    if not _has_table(inspector, "annotation_audit"):
        op.create_table(
            "annotation_audit",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("annotation_id", sa.String(), nullable=False),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("action_at", sa.DateTime(), nullable=False),
            sa.Column("actor", sa.String(), nullable=False),
            sa.Column("notes", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    if _has_table(inspector, "annotation_audit") and not _has_index(
        inspector, "annotation_audit", "ix_annotation_audit_annotation_at"
    ):
        op.create_index(
            "ix_annotation_audit_annotation_at",
            "annotation_audit",
            ["annotation_id", "action_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "annotation_audit"):
        if _has_index(inspector, "annotation_audit", "ix_annotation_audit_annotation_at"):
            op.drop_index("ix_annotation_audit_annotation_at", table_name="annotation_audit")
        op.drop_table("annotation_audit")

    if _has_table(inspector, "user_annotations_v2"):
        if _has_index(inspector, "user_annotations_v2", "ix_user_annotations_window"):
            op.drop_index("ix_user_annotations_window", table_name="user_annotations_v2")
        if _has_index(inspector, "user_annotations_v2", "ix_user_annotations_lookup"):
            op.drop_index("ix_user_annotations_lookup", table_name="user_annotations_v2")
        op.drop_table("user_annotations_v2")

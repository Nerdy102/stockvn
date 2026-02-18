"""add screener v4 saved screens and runs

Revision ID: 20260218_0010
Revises: 20260218_0009
Create Date: 2026-02-18 00:10:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0010"
down_revision = "20260218_0009"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "saved_screens"):
        op.create_table(
            "saved_screens",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("workspace_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("screen_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "saved_screens") and not _has_index(
        inspector, "saved_screens", "ix_saved_screens_workspace_updated"
    ):
        op.create_index(
            "ix_saved_screens_workspace_updated",
            "saved_screens",
            ["workspace_id", "updated_at"],
            unique=False,
        )
    if _has_table(inspector, "saved_screens") and not _has_index(
        inspector, "saved_screens", "ux_saved_screens_workspace_name"
    ):
        op.create_index(
            "ux_saved_screens_workspace_name",
            "saved_screens",
            ["workspace_id", "name"],
            unique=True,
        )

    if not _has_table(inspector, "screen_runs"):
        op.create_table(
            "screen_runs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("saved_screen_id", sa.String(), nullable=True),
            sa.Column("as_of_date", sa.Date(), nullable=False),
            sa.Column("run_at", sa.DateTime(), nullable=False),
            sa.Column("screen_hash", sa.String(), nullable=False),
            sa.Column("universe_hash", sa.String(), nullable=False),
            sa.Column("summary_json", sa.JSON(), nullable=False),
            sa.Column("diff_json", sa.JSON(), nullable=False),
            sa.ForeignKeyConstraint(["saved_screen_id"], ["saved_screens.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "screen_runs") and not _has_index(
        inspector, "screen_runs", "ix_screen_runs_saved_screen_run_at"
    ):
        op.create_index(
            "ix_screen_runs_saved_screen_run_at",
            "screen_runs",
            ["saved_screen_id", "run_at"],
            unique=False,
        )
    if _has_table(inspector, "screen_runs") and not _has_index(
        inspector, "screen_runs", "ix_screen_runs_as_of"
    ):
        op.create_index("ix_screen_runs_as_of", "screen_runs", ["as_of_date"], unique=False)
    if _has_table(inspector, "screen_runs") and not _has_index(
        inspector, "screen_runs", "ux_screen_runs_idempotent"
    ):
        op.create_index(
            "ux_screen_runs_idempotent",
            "screen_runs",
            ["saved_screen_id", "as_of_date", "screen_hash", "universe_hash"],
            unique=True,
        )

    if not _has_table(inspector, "screen_run_items"):
        op.create_table(
            "screen_run_items",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("rank", sa.Integer(), nullable=False),
            sa.Column("score", sa.Float(), nullable=False),
            sa.Column("explain_json", sa.JSON(), nullable=False),
            sa.ForeignKeyConstraint(["run_id"], ["screen_runs.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "screen_run_items") and not _has_index(
        inspector, "screen_run_items", "ix_screen_run_items_run_rank"
    ):
        op.create_index(
            "ix_screen_run_items_run_rank", "screen_run_items", ["run_id", "rank"], unique=False
        )
    if _has_table(inspector, "screen_run_items") and not _has_index(
        inspector, "screen_run_items", "ix_screen_run_items_symbol"
    ):
        op.create_index("ix_screen_run_items_symbol", "screen_run_items", ["symbol"], unique=False)
    if _has_table(inspector, "screen_run_items") and not _has_index(
        inspector, "screen_run_items", "ux_screen_run_items_run_symbol"
    ):
        op.create_index(
            "ux_screen_run_items_run_symbol",
            "screen_run_items",
            ["run_id", "symbol"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "screen_run_items"):
        for idx in [
            "ux_screen_run_items_run_symbol",
            "ix_screen_run_items_symbol",
            "ix_screen_run_items_run_rank",
        ]:
            if _has_index(inspector, "screen_run_items", idx):
                op.drop_index(idx, table_name="screen_run_items")
        op.drop_table("screen_run_items")

    if _has_table(inspector, "screen_runs"):
        for idx in [
            "ux_screen_runs_idempotent",
            "ix_screen_runs_as_of",
            "ix_screen_runs_saved_screen_run_at",
        ]:
            if _has_index(inspector, "screen_runs", idx):
                op.drop_index(idx, table_name="screen_runs")
        op.drop_table("screen_runs")

    if _has_table(inspector, "saved_screens"):
        for idx in ["ux_saved_screens_workspace_name", "ix_saved_screens_workspace_updated"]:
            if _has_index(inspector, "saved_screens", idx):
                op.drop_index(idx, table_name="saved_screens")
        op.drop_table("saved_screens")

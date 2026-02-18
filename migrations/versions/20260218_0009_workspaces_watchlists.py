"""add workspaces watchlists and tag dictionary

Revision ID: 20260218_0009
Revises: 20260218_0008
Create Date: 2026-02-18 00:09:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260218_0009"
down_revision = "20260218_0008"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "workspaces"):
        op.create_table(
            "workspaces",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "workspaces") and not _has_index(
        inspector, "workspaces", "ux_workspaces_user_name"
    ):
        op.create_index("ux_workspaces_user_name", "workspaces", ["user_id", "name"], unique=True)

    if not _has_table(inspector, "watchlists"):
        op.create_table(
            "watchlists",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("workspace_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "watchlists") and not _has_index(
        inspector, "watchlists", "ux_watchlists_workspace_name"
    ):
        op.create_index(
            "ux_watchlists_workspace_name", "watchlists", ["workspace_id", "name"], unique=True
        )

    if not _has_table(inspector, "watchlist_items"):
        op.create_table(
            "watchlist_items",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("watchlist_id", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("tags_json", sa.String(), nullable=False),
            sa.Column("note_text", sa.String(), nullable=False, server_default=""),
            sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["watchlist_id"], ["watchlists.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)
    if _has_table(inspector, "watchlist_items") and not _has_index(
        inspector, "watchlist_items", "ux_watchlist_items_watchlist_symbol"
    ):
        op.create_index(
            "ux_watchlist_items_watchlist_symbol",
            "watchlist_items",
            ["watchlist_id", "symbol"],
            unique=True,
        )
    if _has_table(inspector, "watchlist_items") and not _has_index(
        inspector, "watchlist_items", "idx_watchlist_items_watchlist_id"
    ):
        op.create_index(
            "idx_watchlist_items_watchlist_id", "watchlist_items", ["watchlist_id"], unique=False
        )
    if _has_table(inspector, "watchlist_items") and not _has_index(
        inspector, "watchlist_items", "idx_watchlist_items_symbol"
    ):
        op.create_index("idx_watchlist_items_symbol", "watchlist_items", ["symbol"], unique=False)

    if not _has_table(inspector, "tag_dictionary"):
        op.create_table(
            "tag_dictionary",
            sa.Column("tag", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=False, server_default=""),
            sa.Column("category", sa.String(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("tag"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "tag_dictionary"):
        op.drop_table("tag_dictionary")

    if _has_table(inspector, "watchlist_items"):
        if _has_index(inspector, "watchlist_items", "idx_watchlist_items_symbol"):
            op.drop_index("idx_watchlist_items_symbol", table_name="watchlist_items")
        if _has_index(inspector, "watchlist_items", "idx_watchlist_items_watchlist_id"):
            op.drop_index("idx_watchlist_items_watchlist_id", table_name="watchlist_items")
        if _has_index(inspector, "watchlist_items", "ux_watchlist_items_watchlist_symbol"):
            op.drop_index("ux_watchlist_items_watchlist_symbol", table_name="watchlist_items")
        op.drop_table("watchlist_items")

    if _has_table(inspector, "watchlists"):
        if _has_index(inspector, "watchlists", "ux_watchlists_workspace_name"):
            op.drop_index("ux_watchlists_workspace_name", table_name="watchlists")
        op.drop_table("watchlists")

    if _has_table(inspector, "workspaces"):
        if _has_index(inspector, "workspaces", "ux_workspaces_user_name"):
            op.drop_index("ux_workspaces_user_name", table_name="workspaces")
        op.drop_table("workspaces")

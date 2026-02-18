"""add canonical event log

Revision ID: 20260218_0005
Revises: 20260218_0004
Create Date: 2026-02-18 00:05:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260218_0005"
down_revision = "20260218_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "event_log" not in inspector.get_table_names():
        op.create_table(
            "event_log",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("ts_utc", sa.DateTime(timezone=False), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("event_type", sa.String(), nullable=False),
            sa.Column("symbol", sa.String(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("payload_hash", sa.String(), nullable=False),
            sa.Column("run_id", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("event_log")}
    if "ix_event_log_ts_utc" not in existing_indexes:
        op.create_index("ix_event_log_ts_utc", "event_log", ["ts_utc"], unique=False)
    if "ix_event_log_source_type" not in existing_indexes:
        op.create_index("ix_event_log_source_type", "event_log", ["source", "event_type"], unique=False)
    if "ix_event_log_symbol_ts" not in existing_indexes:
        op.create_index("ix_event_log_symbol_ts", "event_log", ["symbol", "ts_utc"], unique=False)
    if "ix_event_log_run_id" not in existing_indexes:
        op.create_index("ix_event_log_run_id", "event_log", ["run_id"], unique=False)
    if "ix_event_log_payload_hash" not in existing_indexes:
        op.create_index("ix_event_log_payload_hash", "event_log", ["payload_hash"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "event_log" not in inspector.get_table_names():
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("event_log")}
    if "ix_event_log_payload_hash" in existing_indexes:
        op.drop_index("ix_event_log_payload_hash", table_name="event_log")
    if "ix_event_log_run_id" in existing_indexes:
        op.drop_index("ix_event_log_run_id", table_name="event_log")
    if "ix_event_log_symbol_ts" in existing_indexes:
        op.drop_index("ix_event_log_symbol_ts", table_name="event_log")
    if "ix_event_log_source_type" in existing_indexes:
        op.drop_index("ix_event_log_source_type", table_name="event_log")
    if "ix_event_log_ts_utc" in existing_indexes:
        op.drop_index("ix_event_log_ts_utc", table_name="event_log")
    op.drop_table("event_log")

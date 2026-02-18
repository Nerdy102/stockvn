"""add universe PIT tables

Revision ID: 20260218_0004
Revises: 20260218_0003
Create Date: 2026-02-18 00:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260218_0004"
down_revision = "20260218_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ticker_lifecycle",
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("first_trading_date", sa.Date(), nullable=False),
        sa.Column("last_trading_date", sa.Date(), nullable=True),
        sa.Column("exchange", sa.String(), nullable=False),
        sa.Column("sectype", sa.String(), nullable=False),
        sa.Column("sector", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )
    op.create_index(op.f("ix_ticker_lifecycle_exchange"), "ticker_lifecycle", ["exchange"], unique=False)
    op.create_index(op.f("ix_ticker_lifecycle_first_trading_date"), "ticker_lifecycle", ["first_trading_date"], unique=False)
    op.create_index(op.f("ix_ticker_lifecycle_last_trading_date"), "ticker_lifecycle", ["last_trading_date"], unique=False)
    op.create_index(op.f("ix_ticker_lifecycle_sectype"), "ticker_lifecycle", ["sectype"], unique=False)
    op.create_index(op.f("ix_ticker_lifecycle_sector"), "ticker_lifecycle", ["sector"], unique=False)
    op.create_index(op.f("ix_ticker_lifecycle_source"), "ticker_lifecycle", ["source"], unique=False)

    op.create_table(
        "index_membership",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("index_code", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_index_membership_end_date"), "index_membership", ["end_date"], unique=False)
    op.create_index(op.f("ix_index_membership_index_code"), "index_membership", ["index_code"], unique=False)
    op.create_index("ix_index_membership_index_date", "index_membership", ["index_code", "start_date", "end_date"], unique=False)
    op.create_index(op.f("ix_index_membership_source"), "index_membership", ["source"], unique=False)
    op.create_index(op.f("ix_index_membership_start_date"), "index_membership", ["start_date"], unique=False)
    op.create_index(op.f("ix_index_membership_symbol"), "index_membership", ["symbol"], unique=False)

    op.create_table(
        "universe_audit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("universe_name", sa.String(), nullable=False),
        sa.Column("included_count", sa.Integer(), nullable=False),
        sa.Column("excluded_json_breakdown", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_universe_audit_date"), "universe_audit", ["date"], unique=False)
    op.create_index("ix_universe_audit_date_name", "universe_audit", ["date", "universe_name"], unique=True)
    op.create_index(op.f("ix_universe_audit_universe_name"), "universe_audit", ["universe_name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_universe_audit_universe_name"), table_name="universe_audit")
    op.drop_index("ix_universe_audit_date_name", table_name="universe_audit")
    op.drop_index(op.f("ix_universe_audit_date"), table_name="universe_audit")
    op.drop_table("universe_audit")

    op.drop_index(op.f("ix_index_membership_symbol"), table_name="index_membership")
    op.drop_index(op.f("ix_index_membership_start_date"), table_name="index_membership")
    op.drop_index(op.f("ix_index_membership_source"), table_name="index_membership")
    op.drop_index("ix_index_membership_index_date", table_name="index_membership")
    op.drop_index(op.f("ix_index_membership_index_code"), table_name="index_membership")
    op.drop_index(op.f("ix_index_membership_end_date"), table_name="index_membership")
    op.drop_table("index_membership")

    op.drop_index(op.f("ix_ticker_lifecycle_source"), table_name="ticker_lifecycle")
    op.drop_index(op.f("ix_ticker_lifecycle_sector"), table_name="ticker_lifecycle")
    op.drop_index(op.f("ix_ticker_lifecycle_sectype"), table_name="ticker_lifecycle")
    op.drop_index(op.f("ix_ticker_lifecycle_last_trading_date"), table_name="ticker_lifecycle")
    op.drop_index(op.f("ix_ticker_lifecycle_first_trading_date"), table_name="ticker_lifecycle")
    op.drop_index(op.f("ix_ticker_lifecycle_exchange"), table_name="ticker_lifecycle")
    op.drop_table("ticker_lifecycle")

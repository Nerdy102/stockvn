"""initial schema with postgres partitions

Revision ID: 20260218_0001
Revises:
Create Date: 2026-02-18 00:00:00
"""

from __future__ import annotations

import datetime as dt

import sqlalchemy as sa
from alembic import op
from core.db.models import SQLModel

# revision identifiers, used by Alembic.
revision = "20260218_0001"
down_revision = None
branch_labels = None
depends_on = None

PARTITIONED_TABLES = {"prices_ohlcv", "quotes_l2", "trades_tape"}
FUTURE_TABLES_EXCLUDED_FROM_INITIAL = {
    "daily_flow_features",
    "daily_orderbook_features",
    "daily_intraday_features",
    "feature_last_processed",
}


def _month_floor(value: dt.datetime) -> dt.datetime:
    return dt.datetime(value.year, value.month, 1)


def _next_month(value: dt.datetime) -> dt.datetime:
    if value.month == 12:
        return dt.datetime(value.year + 1, 1, 1)
    return dt.datetime(value.year, value.month + 1, 1)


def _create_month_partition(table_name: str, month_start: dt.datetime) -> None:
    month_end = _next_month(month_start)
    suffix = month_start.strftime("%Y%m")
    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name}_{suffix}
            PARTITION OF {table_name}
            FOR VALUES FROM (:start_ts) TO (:end_ts)
            """
        ).bindparams(start_ts=month_start, end_ts=month_end)
    )


def _create_parent_partitioned_tables() -> None:
    op.execute(
        """
        CREATE TABLE prices_ohlcv (
            symbol VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            ts_utc TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            open FLOAT NOT NULL,
            high FLOAT NOT NULL,
            low FLOAT NOT NULL,
            close FLOAT NOT NULL,
            volume FLOAT NOT NULL,
            value_vnd FLOAT NOT NULL DEFAULT 0.0,
            source VARCHAR NOT NULL,
            quality_flags JSON NOT NULL,
            PRIMARY KEY (symbol, timeframe, ts_utc)
        ) PARTITION BY RANGE (ts_utc)
        """
    )
    op.execute(
        """
        CREATE TABLE quotes_l2 (
            id SERIAL NOT NULL,
            symbol VARCHAR NOT NULL,
            ts_utc TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            bids JSON NOT NULL,
            asks JSON NOT NULL,
            source VARCHAR NOT NULL,
            PRIMARY KEY (id, ts_utc),
            CONSTRAINT ux_quotes_l2_symbol_ts_source UNIQUE (symbol, ts_utc, source)
        ) PARTITION BY RANGE (ts_utc)
        """
    )
    op.execute(
        """
        CREATE TABLE trades_tape (
            id SERIAL NOT NULL,
            symbol VARCHAR NOT NULL,
            ts_utc TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            last_price FLOAT NOT NULL,
            last_vol FLOAT NOT NULL,
            side VARCHAR,
            source VARCHAR NOT NULL,
            PRIMARY KEY (id, ts_utc),
            CONSTRAINT ux_trades_tape_symbol_ts_source UNIQUE (symbol, ts_utc, source)
        ) PARTITION BY RANGE (ts_utc)
        """
    )


def _create_partitions_window() -> None:
    now = dt.datetime.utcnow()
    current = _month_floor(now)
    for offset in range(-24, 4):
        month = current
        if offset >= 0:
            for _ in range(offset):
                month = _next_month(month)
        else:
            for _ in range(abs(offset)):
                year = month.year
                mon = month.month - 1
                if mon == 0:
                    mon = 12
                    year -= 1
                month = dt.datetime(year, mon, 1)
        _create_month_partition("prices_ohlcv", month)
        _create_month_partition("quotes_l2", month)
        _create_month_partition("trades_tape", month)


def _create_indexes_postgres_only() -> None:
    op.create_index(
        "ix_prices_ohlcv_symbol_timeframe_ts_utc",
        "prices_ohlcv",
        ["symbol", "timeframe", "ts_utc"],
        unique=False,
    )
    op.create_index("ix_prices_timeframe_timestamp", "prices_ohlcv", ["timeframe", "ts_utc"])
    op.create_index("ix_prices_timestamp", "prices_ohlcv", ["ts_utc"])

    op.create_index("ix_quotes_l2_symbol_ts_utc", "quotes_l2", ["symbol", "ts_utc"])

    op.create_index("ix_trades_tape_symbol_ts_utc", "trades_tape", ["symbol", "ts_utc"])


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        _create_parent_partitioned_tables()

    tables = [
        table
        for name, table in SQLModel.metadata.tables.items()
        if not (dialect == "postgresql" and name in PARTITIONED_TABLES)
        and name not in FUTURE_TABLES_EXCLUDED_FROM_INITIAL
    ]
    SQLModel.metadata.create_all(bind=bind, tables=tables)

    if dialect == "postgresql":
        _create_partitions_window()

    if dialect == "postgresql":
        _create_indexes_postgres_only()


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        for table in ("trades_tape", "quotes_l2", "prices_ohlcv"):
            op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    tables = [
        table
        for name, table in reversed(list(SQLModel.metadata.tables.items()))
        if not (dialect == "postgresql" and name in PARTITIONED_TABLES)
        and name not in FUTURE_TABLES_EXCLUDED_FROM_INITIAL
    ]
    for table in tables:
        op.drop_table(table.name)

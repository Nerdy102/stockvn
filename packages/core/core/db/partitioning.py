from __future__ import annotations

import datetime as dt

import sqlalchemy as sa
from sqlmodel import Session

PARTITIONED_TABLES = ("prices_ohlcv", "quotes_l2", "trades_tape")


def _month_start(value: dt.datetime) -> dt.datetime:
    return dt.datetime(value.year, value.month, 1)


def _add_months(value: dt.datetime, months: int) -> dt.datetime:
    year = value.year + ((value.month - 1 + months) // 12)
    month = ((value.month - 1 + months) % 12) + 1
    return dt.datetime(year, month, 1)


def _partition_name(table: str, month_start: dt.datetime) -> str:
    return f"{table}_{month_start.strftime('%Y%m')}"


def create_month_partition(session: Session, table_name: str, month_start: dt.datetime) -> None:
    month_start = _month_start(month_start)
    month_end = _add_months(month_start, 1)
    part_name = _partition_name(table_name, month_start)
    session.exec(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {part_name}
            PARTITION OF {table_name}
            FOR VALUES FROM (:start_ts) TO (:end_ts)
            """
        ).bindparams(start_ts=month_start, end_ts=month_end)
    )


def ensure_partitions_monthly(session: Session, months_ahead: int = 3) -> int:
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return 0

    created = 0
    base_month = _month_start(dt.datetime.utcnow())
    for table_name in PARTITIONED_TABLES:
        for month_delta in range(1, months_ahead + 1):
            month_start = _add_months(base_month, month_delta)
            part_name = _partition_name(table_name, month_start)
            exists = session.exec(
                sa.text("SELECT to_regclass(:name)").bindparams(name=part_name)
            ).scalar_one()
            if exists is None:
                create_month_partition(session, table_name, month_start)
                created += 1
    session.commit()
    return created

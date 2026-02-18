from __future__ import annotations

import datetime as dt
import os

import pytest
from core.db.partitioning import _add_months
from core.db.session import get_engine
from sqlalchemy import text
from sqlmodel import Session
from worker_scheduler.jobs import ensure_partitions_monthly


@pytest.mark.postgres
def test_partition_manager_creates_next_month_partition() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")

    engine = get_engine(database_url)
    next_month = _add_months(
        dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0), 3
    )
    part_name = f"prices_ohlcv_{next_month.strftime('%Y%m')}"

    with Session(engine) as session:
        session.exec(text(f"DROP TABLE IF EXISTS {part_name}"))
        session.commit()

        created = ensure_partitions_monthly(session)
        assert created >= 1

        exists = session.exec(
            text("SELECT to_regclass(:name)").bindparams(name=part_name)
        ).scalar_one()
        assert exists == part_name

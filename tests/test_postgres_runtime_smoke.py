from __future__ import annotations

import os

import pytest
from core.db.partitioning import ensure_partitions_monthly
from core.db.session import get_engine
from sqlmodel import Session


@pytest.mark.postgres
def test_worker_runtime_smoke_postgres() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")

    engine = get_engine(database_url)
    with Session(engine) as session:
        created = ensure_partitions_monthly(session)
    assert created >= 0

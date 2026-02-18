from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, text


@pytest.mark.postgres
def test_alembic_upgrade_head_postgres() -> None:
    if importlib.util.find_spec("alembic") is None:
        pytest.skip("alembic is not installed")

    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")

    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env["PYTHONPATH"] = (
        "packages/core:packages/data:services/api_fastapi:services/worker_scheduler:apps"
    )

    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env)

    engine = create_engine(database_url)
    with engine.connect() as conn:
        partitioned = (
            conn.execute(
                text(
                    """
                SELECT relname
                FROM pg_class c
                JOIN pg_partitioned_table p ON p.partrelid = c.oid
                WHERE relname IN ('prices_ohlcv', 'quotes_l2', 'trades_tape')
                """
                )
            )
            .scalars()
            .all()
        )

        assert set(partitioned) == {"prices_ohlcv", "quotes_l2", "trades_tape"}

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_head_sqlite(tmp_path) -> None:
    if importlib.util.find_spec("alembic") is None:
        pytest.skip("alembic is not installed")

    db_path = tmp_path / "test_sqlite.db"
    database_url = f"sqlite:///{db_path}"
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    env["PYTHONPATH"] = (
        "packages/core:packages/data:services/api_fastapi:services/worker_scheduler:apps"
    )

    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env)

    engine = create_engine(database_url)
    inspector = inspect(engine)

    assert "prices_ohlcv" in inspector.get_table_names()
    assert "quotes_l2" in inspector.get_table_names()
    assert "trades_tape" in inspector.get_table_names()

    prices_indexes = {idx["name"] for idx in inspector.get_indexes("prices_ohlcv")}
    assert "ix_prices_ohlcv_symbol_timeframe_ts_utc" in prices_indexes

    quote_indexes = {idx["name"] for idx in inspector.get_indexes("quotes_l2")}
    assert "ix_quotes_l2_symbol_ts_utc" in quote_indexes

    trade_indexes = {idx["name"] for idx in inspector.get_indexes("trades_tape")}
    assert "ix_trades_tape_symbol_ts_utc" in trade_indexes

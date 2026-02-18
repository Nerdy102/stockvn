from __future__ import annotations

import datetime as dt
import os
import time
import tracemalloc
from pathlib import Path

from core.data_lake.feature_source import load_table_df
from core.data_lake.parquet_export import export_partitioned_parquet_for_day
from core.db.models import PriceOHLCV, Ticker
from core.db.session import get_engine
from core.settings import Settings
from sqlmodel import SQLModel, Session


def test_parquet_or_db_query_perf_smoke_small_ci(tmp_path: Path) -> None:
    budget_sec = float(os.getenv("PARQUET_QUERY_PERF_BUDGET_SEC", "2.5"))
    mem_budget_mb = float(os.getenv("PARQUET_QUERY_MEM_BUDGET_MB", "512"))

    db_path = tmp_path / "perf.db"
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    start = dt.date(2023, 1, 1)
    symbols = [f"S{i:03d}" for i in range(80)]
    days = [start + dt.timedelta(days=i) for i in range(140)]

    with Session(engine) as s:
        for sym in symbols:
            s.add(Ticker(symbol=sym, name=sym, exchange="HOSE", sector="T", industry="I"))
            for i, day in enumerate(days):
                c = 10.0 + i * 0.01
                s.add(
                    PriceOHLCV(
                        symbol=sym,
                        timeframe="1D",
                        timestamp=dt.datetime.combine(day, dt.time(0, 0)),
                        open=c,
                        high=c * 1.01,
                        low=c * 0.99,
                        close=c,
                        volume=1000 + i,
                        value_vnd=(1000 + i) * c,
                    )
                )
        s.commit()

        settings = Settings(DATABASE_URL=f"sqlite:///{db_path}", PARQUET_LAKE_ROOT=str(tmp_path / "lake"))
        for day in days[-30:]:
            export_partitioned_parquet_for_day(s, settings=settings, as_of_date=day)

        tracemalloc.start()
        t0 = time.perf_counter()
        out = load_table_df(
            s,
            model=PriceOHLCV,
            table_name="prices_ohlcv",
            date_col="timestamp",
            settings=settings,
            start_date=days[-30],
            end_date=days[-1],
            symbols=symbols[:40],
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert not out.empty
    assert elapsed <= budget_sec
    assert (peak / (1024 * 1024)) <= mem_budget_mb

from __future__ import annotations

import datetime as dt
import time

from core.db.models import PriceOHLCV
from sqlmodel import Session, SQLModel, create_engine
from worker_scheduler.jobs import compute_indicators_incremental


def test_performance_smoke_small_ci() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    start = time.perf_counter()
    with Session(engine) as s:
        base = dt.datetime(2020, 1, 1)
        for sym_i in range(30):
            sym = f"S{sym_i:04d}"
            for d in range(200):
                ts = base + dt.timedelta(days=d)
                s.merge(
                    PriceOHLCV(
                        symbol=sym,
                        timeframe="1D",
                        timestamp=ts,
                        open=10,
                        high=11,
                        low=9,
                        close=10 + (d % 7) * 0.1,
                        volume=1000 + d,
                        value_vnd=10000,
                    )
                )
        s.commit()
        ingest_elapsed = time.perf_counter() - start

        c0 = time.perf_counter()
        updates = compute_indicators_incremental(s)
        comp_elapsed = time.perf_counter() - c0

    assert updates > 0
    assert ingest_elapsed < 35
    assert comp_elapsed < 20

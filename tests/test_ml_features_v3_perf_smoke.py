from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from core.alpha_v3.features import build_ml_features_v3


def test_ml_features_v3_memory_perf_smoke() -> None:
    max_seconds = float(os.getenv("ML_FEATURES_V3_PERF_BUDGET_SEC", "8.0"))
    symbols = [f"S{i:03d}" for i in range(200)]
    start = dt.date(2023, 1, 1)
    days = [start + dt.timedelta(days=i) for i in range(500)]

    rows = []
    for sym in symbols:
        for i, day in enumerate(days):
            close = 10.0 + (i * 0.01)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": day.isoformat(),
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000 + i,
                    "value_vnd": close * (1000 + i),
                }
            )
    # Add VNINDEX required by regime builder.
    for i, day in enumerate(days):
        close = 1000.0 + i
        rows.append(
            {
                "symbol": "VNINDEX",
                "timestamp": day.isoformat(),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000,
                "value_vnd": close * 1_000_000,
            }
        )

    prices = pd.DataFrame(rows)

    t0 = time.perf_counter()
    out = build_ml_features_v3(prices=prices)
    elapsed = time.perf_counter() - t0

    assert len(out) == len(prices)
    assert elapsed <= max_seconds

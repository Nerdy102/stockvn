from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from core.ml.features import build_ml_features


def test_performance_smoke_ml_features_200x500() -> None:
    rows = []
    base = dt.datetime(2020, 1, 1)
    for si in range(200):
        sym = f"S{si:04d}"
        for d in range(500):
            ts = base + dt.timedelta(days=d)
            px = 10 + (d % 37) * 0.1 + si * 0.001
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "close": px,
                    "volume": 1_000_000 + d,
                    "value_vnd": px * (1_000_000 + d),
                    "exchange": "HOSE",
                    "sector": "TECH",
                }
            )
    df = pd.DataFrame(rows)

    t0 = time.perf_counter()
    feat = build_ml_features(df)
    elapsed = time.perf_counter() - t0

    assert not feat.empty
    assert elapsed < 8.0

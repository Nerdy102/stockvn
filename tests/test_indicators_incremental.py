from __future__ import annotations

import numpy as np
import pandas as pd
from core.indicators import RSIState, ema, ema_incremental, rsi, rsi_incremental


def test_ema_incremental_matches_batch() -> None:
    prices = np.linspace(10, 20, 200)
    batch = ema(pd.Series(prices), span=20)

    prev = None
    out = []
    for p in prices:
        prev = ema_incremental(float(p), prev, span=20)
        out.append(prev)

    assert np.allclose(batch.values, np.array(out), atol=1e-9)


def test_rsi_incremental_stable() -> None:
    prices = pd.Series([100 + (i % 5) for i in range(250)], dtype=float)
    batch = rsi(prices, window=14).fillna(0)

    state = RSIState()
    out = []
    for p in prices:
        v, state = rsi_incremental(float(p), state, window=14)
        out.append(v)

    assert out[-1] >= 0
    assert out[-1] <= 100
    assert abs(out[-1] - float(batch.iloc[-1])) < 5.0

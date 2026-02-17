from __future__ import annotations

import random

from core.indicators import add_indicators
from data.adapters.ssi_mapper import SSIMapper


def test_random_ohlcv_indicator_no_crash() -> None:
    import pandas as pd

    rng = random.Random(42)
    rows = []
    px = 100.0
    for _ in range(500):
        move = rng.uniform(-1.5, 1.5)
        op = px
        cl = max(0.01, px + move)
        hi = max(op, cl) + rng.uniform(0, 1)
        lo = max(0.01, min(op, cl) - rng.uniform(0, 1))
        vol = rng.uniform(1, 1_000_000)
        rows.append({"open": op, "high": hi, "low": lo, "close": cl, "volume": vol})
        px = cl

    out = add_indicators(pd.DataFrame(rows))
    assert "RSI14" in out.columns


def test_random_daily_stock_price_strings_parser() -> None:
    rng = random.Random(1)
    for _ in range(100):
        close = str(round(rng.uniform(1, 200), 2))
        payload = {
            "Tradingdate": "10/01/2025",
            "Symbol": "AAA",
            "Openprice": close,
            "Highestprice": close,
            "Lowestprice": close,
            "Closeprice": close,
            "Totalmatchvol": str(rng.randint(1, 10_000)),
            "Totalmatchval": str(rng.randint(1000, 1000000)),
            "Toreignsellvaltotal": "1234",
            "Netforeivol": "12",
            "Time": "15:00:00",
        }
        bar = SSIMapper.map_daily_stock_price(payload)
        assert bar.close >= 0

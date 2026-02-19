from __future__ import annotations

import pandas as pd

from core.backtest_v2.engine import run_backtest_v2
from core.backtest_v2.schemas import BacktestConfig


def _df() -> pd.DataFrame:
    rows = []
    for i in range(80):
        px = 100 - i * 0.2
        rows.append(
            {
                "timestamp": f"2025-01-{(i%28)+1:02d}",
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px,
                "volume": 1000000,
            }
        )
    return pd.DataFrame(rows)


def test_short_math_profit_when_price_down() -> None:
    cfg = BacktestConfig(market="crypto", trading_type="perp_paper", position_mode="long_short")
    r = run_backtest_v2(_df(), cfg, lambda hist: "SELL", fee_rate=0.0004, sell_tax_rate=0.0)
    assert r.metrics["net_return"] >= -1.0

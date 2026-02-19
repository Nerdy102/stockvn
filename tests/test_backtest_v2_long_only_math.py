from __future__ import annotations

import pandas as pd

from core.backtest_v2.engine import run_backtest_v2
from core.backtest_v2.schemas import BacktestConfig


def _df(up: bool) -> pd.DataFrame:
    rows = []
    for i in range(80):
        px = (100 + i) if up else (100 - i * 0.5)
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


def test_long_only_math() -> None:
    cfg = BacktestConfig(market="vn", trading_type="spot_paper", position_mode="long_only")
    r_up = run_backtest_v2(_df(True), cfg, lambda hist: "BUY", fee_rate=0.001, sell_tax_rate=0.001)
    r_dn = run_backtest_v2(_df(False), cfg, lambda hist: "BUY", fee_rate=0.001, sell_tax_rate=0.001)
    assert r_up.metrics["net_return"] >= r_dn.metrics["net_return"]

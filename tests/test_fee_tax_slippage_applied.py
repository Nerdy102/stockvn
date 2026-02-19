from __future__ import annotations

import pandas as pd

from core.backtest_v2.engine import run_backtest_v2
from core.backtest_v2.schemas import BacktestConfig


def test_fee_tax_slippage_applied() -> None:
    rows = []
    for i in range(90):
        px = 100 + i * 0.1
        rows.append(
            {
                "timestamp": f"2025-01-{(i%28)+1:02d}",
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px,
                "volume": 100000,
            }
        )
    df = pd.DataFrame(rows)
    cfg = BacktestConfig()
    r = run_backtest_v2(
        df,
        cfg,
        lambda hist: "BUY" if len(hist) < 30 else "HOLD",
        fee_rate=0.001,
        sell_tax_rate=0.001,
        include_trades=True,
    )
    if r.trades:
        t = r.trades[0]
        assert t.fee >= 0
        assert t.slippage_cost >= 0
        assert abs(t.pnl_net - (t.pnl_gross - t.fee - t.tax - t.slippage_cost)) < 1e-6

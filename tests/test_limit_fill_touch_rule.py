from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, OrderIntent, _simulate_fill


def test_limit_fill_touch_rule() -> None:
    row = pd.Series({"timestamp": "2025-01-01", "open": 100, "high": 103, "low": 98, "close": 101, "volume": 10000})
    f = _simulate_fill(
        OrderIntent(type="limit", side="BUY", qty=100, limit_price=99),
        row,
        atr_pct=0.01,
        next_open=None,
        cfg=BacktestV3Config(order_type="limit"),
        fee_cfg={"commission_rate": 0.001, "slippage_model": {}},
        tax_rate=0.0,
    )
    assert f.qty > 0

    f2 = _simulate_fill(
        OrderIntent(type="limit", side="BUY", qty=100, limit_price=97),
        row,
        atr_pct=0.01,
        next_open=None,
        cfg=BacktestV3Config(order_type="limit"),
        fee_cfg={"commission_rate": 0.001, "slippage_model": {}},
        tax_rate=0.0,
    )
    assert f2.qty == 0

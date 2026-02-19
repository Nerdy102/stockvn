from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, OrderIntent, _simulate_fill


def test_fee_tax_slippage_applied_per_fill() -> None:
    row = pd.Series({"timestamp": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100000})
    fill = _simulate_fill(
        OrderIntent(type="market", side="SELL", qty=100),
        row,
        atr_pct=0.01,
        next_open=None,
        cfg=BacktestV3Config(market="vn"),
        fee_cfg={"commission_rate": 0.001, "slippage_model": {"base_bps": 3, "k_atr": 0.03, "k_part": 50}},
        tax_rate=0.001,
    )
    assert fill.fee > 0
    assert fill.tax > 0
    assert fill.slippage_cost > 0

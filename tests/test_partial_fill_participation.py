from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, OrderIntent, _simulate_fill


def test_partial_fill_participation() -> None:
    row = pd.Series({"timestamp": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    cfg = BacktestV3Config(participation_rate=0.05)
    f = _simulate_fill(
        OrderIntent(type="market", side="BUY", qty=200),
        row,
        atr_pct=0.01,
        next_open=None,
        cfg=cfg,
        fee_cfg={"commission_rate": 0.001, "slippage_model": {}},
        tax_rate=0.0,
    )
    assert f.qty <= 50
    assert f.status == "PARTIAL"

from __future__ import annotations

import pandas as pd

from core.backtest_v2.engine import run_backtest_v2
from core.backtest_v2.schemas import BacktestConfig


def test_determinism_report_id() -> None:
    rows = []
    for i in range(80):
        px = 100 + i * 0.2
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
    df = pd.DataFrame(rows)
    cfg = BacktestConfig()
    r1 = run_backtest_v2(df, cfg, lambda hist: "BUY", fee_rate=0.001, sell_tax_rate=0.001)
    r2 = run_backtest_v2(df, cfg, lambda hist: "BUY", fee_rate=0.001, sell_tax_rate=0.001)
    assert r1.report_id == r2.report_id
    assert r1.config_hash == r2.config_hash
    assert r1.dataset_hash == r2.dataset_hash

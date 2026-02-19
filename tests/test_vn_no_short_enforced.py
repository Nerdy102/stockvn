from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, run_backtest_v3


def test_vn_no_short_enforced() -> None:
    rows = []
    for i in range(30):
        c = 100 + i
        rows.append({"date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 100000})
    df = pd.DataFrame(rows)

    out = run_backtest_v3(
        df=df,
        symbol="FPT",
        timeframe="1D",
        config=BacktestV3Config(market="vn", position_mode="long_only"),
        signal_fn=lambda _hist: "GIAM",
        fees_taxes_path="configs/fees_taxes.yaml",
        fees_crypto_path="configs/fees_crypto.yaml",
        execution_model_path="configs/execution_model.yaml",
    )
    assert out.exposure_short_pct == 0

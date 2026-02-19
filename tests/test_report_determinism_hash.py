from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, run_backtest_v3


def test_report_determinism_hash() -> None:
    rows = []
    for i in range(40):
        c = 100 + i * 0.5
        rows.append({"date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 100000})
    df = pd.DataFrame(rows)
    cfg = BacktestV3Config()
    f = lambda _h: "TRUNG_TINH"
    r1 = run_backtest_v3(df=df, symbol="FPT", timeframe="1D", config=cfg, signal_fn=f, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml")
    r2 = run_backtest_v3(df=df, symbol="FPT", timeframe="1D", config=cfg, signal_fn=f, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml")
    assert r1.report_id == r2.report_id
    assert r1.dataset_hash == r2.dataset_hash

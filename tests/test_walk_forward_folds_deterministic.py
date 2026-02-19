from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config
from core.validation.walk_forward_v2 import run_walk_forward_v2


def test_walk_forward_folds_deterministic() -> None:
    rows = []
    for i in range(500):
        c = 100 + i * 0.1
        rows.append({"date": str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 100000})
    df = pd.DataFrame(rows)
    cfg = BacktestV3Config()
    f = lambda _h: "TRUNG_TINH"
    r1 = run_walk_forward_v2(df=df, symbol="FPT", timeframe="1D", config=cfg, signal_fn=f, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml")
    r2 = run_walk_forward_v2(df=df, symbol="FPT", timeframe="1D", config=cfg, signal_fn=f, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml")
    assert r1["stability_score"] == r2["stability_score"]
    assert len(r1["per_fold_metrics"]) <= 10

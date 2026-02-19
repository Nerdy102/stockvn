from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config
from core.validation.stress_suite import run_stress_suite


def test_stress_suite_scenarios_fixed() -> None:
    rows = []
    for i in range(300):
        c = 100 + i * 0.2
        rows.append({"date": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 200000})
    df = pd.DataFrame(rows)
    out = run_stress_suite(df=df, symbol="FPT", timeframe="1D", config=BacktestV3Config(), signal_fn=lambda _h: "TRUNG_TINH", fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml")
    # 9 + 3 + 3 + 2 = 17 scenario rows cho non-perp
    assert len(out["stress_table"]) == 17

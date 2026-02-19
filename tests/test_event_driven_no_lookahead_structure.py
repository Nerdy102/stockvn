from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, run_backtest_v3


def _df(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        c = 100 + i * 0.2
        rows.append({"date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 100000})
    return pd.DataFrame(rows)


def test_event_driven_no_lookahead_structure() -> None:
    seen: list[int] = []

    def signal_fn(hist: pd.DataFrame) -> str:
        seen.append(len(hist))
        return "TRUNG_TINH"

    run_backtest_v3(
        df=_df(),
        symbol="FPT",
        timeframe="1D",
        config=BacktestV3Config(),
        signal_fn=signal_fn,
        fees_taxes_path="configs/fees_taxes.yaml",
        fees_crypto_path="configs/fees_crypto.yaml",
        execution_model_path="configs/execution_model.yaml",
    )
    assert seen[0] == 1
    assert seen[-1] == 80
    assert seen == sorted(seen)

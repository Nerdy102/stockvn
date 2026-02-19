from __future__ import annotations

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, FundingRatePoint, run_backtest_v3


def _df() -> pd.DataFrame:
    rows = []
    for i in range(20):
        c = 100 + i
        rows.append({"date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(hours=8*i)), "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1_000_000})
    return pd.DataFrame(rows)


def test_crypto_funding_sign() -> None:
    rates = [FundingRatePoint(ts=str(pd.Timestamp("2025-01-01") + pd.Timedelta(hours=8*i)), symbol="BTCUSDT", funding_rate=0.001) for i in range(20)]

    long_sig = lambda _hist: "TANG"
    r_long = run_backtest_v3(df=_df(), symbol="BTCUSDT", timeframe="60m", config=BacktestV3Config(market="crypto", trading_type="perp_paper", position_mode="long_short"), signal_fn=long_sig, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml", funding_rates=rates)

    short_sig = lambda _hist: "GIAM"
    r_short = run_backtest_v3(df=_df(), symbol="BTCUSDT", timeframe="60m", config=BacktestV3Config(market="crypto", trading_type="perp_paper", position_mode="long_short"), signal_fn=short_sig, fees_taxes_path="configs/fees_taxes.yaml", fees_crypto_path="configs/fees_crypto.yaml", execution_model_path="configs/execution_model.yaml", funding_rates=rates)

    assert r_long.costs_breakdown["funding_total"] <= 0
    assert r_short.costs_breakdown["funding_total"] >= 0

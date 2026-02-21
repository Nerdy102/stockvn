from __future__ import annotations

import pandas as pd

from research.evaluate_model_vn10 import build_features, run_walk_forward_backtest


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=80)
    rows = []
    for i, date in enumerate(dates):
        for symbol, base in [("AAA", 100.0), ("BBB", 200.0)]:
            close = base + i + (5.0 if symbol == "AAA" else -3.0)
            open_px = close * (1.0 - 0.001)
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": open_px,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + i,
                    "value_vnd": close * (1_000_000 + i),
                }
            )
    return pd.DataFrame(rows)


def test_feature_does_not_use_future_close() -> None:
    prices = _synthetic_prices()
    features = build_features(prices)
    row = features[
        (features["symbol"] == "AAA") & (features["date"] == pd.Timestamp("2024-02-05"))
    ].iloc[0]

    symbol_hist = prices[prices["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
    idx = symbol_hist.index[symbol_hist["date"] == pd.Timestamp("2024-02-05")][0]
    expected_mom5 = symbol_hist.loc[idx, "close"] / symbol_hist.loc[idx - 5, "close"] - 1.0

    assert abs(float(row["mom5"]) - float(expected_mom5)) < 1e-12


def test_backtest_executes_t_plus_1() -> None:
    prices = _synthetic_prices()
    features = build_features(prices)
    equity, trades, _ = run_walk_forward_backtest(
        features,
        train_end=None,
        top_k=1,
        min_score=-999.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    assert not equity.empty
    assert not trades.empty

    dates = sorted(features["date"].unique().tolist())
    first_trade = trades.iloc[0]
    signal_date = pd.Timestamp(first_trade["signal_date"])
    exec_date = pd.Timestamp(first_trade["date"])
    idx = dates.index(signal_date)

    assert exec_date == dates[idx + 1]

import numpy as np
import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.market_rules import load_market_rules


def test_backtest_outputs_metrics_and_equity_curve() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="D"),
            "open": [10000, 10050, 10100, 10080, 10120, 10090, 10150, 10200],
            "high": [10100, 10150, 10200, 10150, 10200, 10180, 10250, 10300],
            "low": [9900, 10000, 10050, 10020, 10070, 10040, 10100, 10150],
            "close": [10050, 10100, 10080, 10120, 10090, 10150, 10200, 10180],
            "value_vnd": [7e9] * 8,
            "atr14": [120] * 8,
            "ceiling_price": [10700] * 8,
            "floor_price": [9300] * 8,
        }
    )
    signal = pd.Series([1, 1, 0, 1, 0, 1, 0, 0], index=pd.to_datetime(bars["timestamp"]))

    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=300_000_000.0),
    )

    assert not out["backtest_runs"].empty
    assert not out["backtest_metrics"].empty
    assert set(out["backtest_metrics"].columns) == {"run_hash", "metric_name", "metric_value"}
    assert np.isfinite(out["backtest_metrics"]["metric_value"].to_numpy(dtype=float)).all()
    assert not out["backtest_equity_curve"].empty

    metrics = out["metrics"]
    vals = np.array(list(metrics.values()), dtype=float)
    assert np.isfinite(vals).all()

    out2 = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=300_000_000.0),
    )
    pd.testing.assert_frame_equal(out["equity_curve"], out2["equity_curve"])
    assert out["backtest_runs"].iloc[0]["run_hash"] == out2["backtest_runs"].iloc[0]["run_hash"]


def test_next_bar_execution_dates() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [10000, 10100, 10200],
            "high": [10100, 10200, 10300],
            "low": [9900, 10000, 10100],
            "close": [10050, 10150, 10250],
            "value_vnd": [6e9, 6e9, 6e9],
            "atr14": [100, 100, 100],
            "ceiling_price": [10700, 10800, 10900],
            "floor_price": [9300, 9400, 9500],
        }
    )
    signal = pd.Series([1, 0, 0], index=pd.to_datetime(bars["timestamp"]))
    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=100_000_000.0),
    )
    first_trade = out["trades"].iloc[0]
    assert str(first_trade["signal_date"]) == "2025-01-01"
    assert str(first_trade["date"]) == "2025-01-02"

import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.market_rules import load_market_rules


def test_fill_penalty_reduces_filled_qty() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [10000, 10000, 10000],
            "high": [10100, 10100, 10100],
            "low": [9900, 9900, 9900],
            "close": [10700, 10000, 10000],
            "value_vnd": [10e9, 10e9, 10e9],
            "atr14": [100, 100, 100],
            "ceiling_price": [10700, 10700, 10700],
            "floor_price": [9300, 9300, 9300],
        }
    )
    signal = pd.Series([1, 1, 1], index=bars.index)
    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=100_000_000.0),
    )

    t0 = out["trades"].iloc[0]
    assert t0["side"] == "BUY"
    assert t0["fill_ratio"] == 0.2
    assert t0["filled_qty"] < t0["order_qty"]

import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.market_rules import load_market_rules


def test_sell_clamp_bug_fee_tax_regression() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "open": [10000, 10000, 12000, 11000, 10900],
            "high": [10100, 10100, 12100, 11100, 11000],
            "low": [9900, 9900, 11900, 10900, 10800],
            "close": [10000, 10000, 11950, 10950, 10850],
            "value_vnd": [2e9] * 5,
            "atr14": [100] * 5,
            "ceiling_price": [10700] * 5,
            "floor_price": [9300, 9300, 9300, 10950, 9300],
        }
    )
    # buy then sell where prior day is floor -> partial fill on sell
    signal = pd.Series([1, 1, 0, 0, 0], index=bars.index)

    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=100_000_000.0),
    )
    trades = out["trades"]
    sells = trades[trades["side"] == "SELL"]
    assert not sells.empty
    s = sells.iloc[0]
    assert s["filled_qty"] <= s["order_qty"]
    assert s["sell_tax"] > 0
    assert s["commission"] > 0

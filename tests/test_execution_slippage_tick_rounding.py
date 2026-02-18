import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.cost_model import calc_slippage_bps
from core.market_rules import load_market_rules


def test_execution_slippage_tick_rounding() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [10000, 10123, 10200],
            "high": [10100, 10300, 10300],
            "low": [9900, 10000, 10100],
            "close": [10050, 10200, 10250],
            "value_vnd": [5e9, 5e9, 5e9],
            "atr14": [200, 250, 250],
            "ceiling_price": [11000, 11100, 11200],
            "floor_price": [9000, 9100, 9200],
        }
    )
    signal = pd.Series([1, 1, 1], index=bars.index)
    rules = load_market_rules("configs/market_rules_vn.yaml")
    cfg = BacktestV3Config(initial_cash=100_000_000.0)

    out = run_backtest_v3(bars, signal, rules, cfg)
    trades = out["trades"]
    first = trades.iloc[0]

    raw_bps = calc_slippage_bps(
        order_notional=first["order_qty"] * bars.iloc[1]["open"],
        adtv=bars.iloc[0]["value_vnd"],
        atr14=bars.iloc[0]["atr14"],
        close=bars.iloc[0]["close"],
        cfg=cfg.slippage,
    )
    raw_exec = bars.iloc[1]["open"] * (1 + raw_bps / 10000.0)
    rounded = rules.round_price(raw_exec, direction="up")

    assert first["side"] == "BUY"
    assert first["exec_price"] == rounded
    assert first["exec_price"] >= raw_exec

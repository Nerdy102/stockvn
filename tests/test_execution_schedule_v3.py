import numpy as np
import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, _ac_weights, _allocate_board_lot, run_backtest_v3
from core.market_rules import load_market_rules


def test_ac_schedule_sums_exactly_and_nonnegative() -> None:
    weights = _ac_weights(3)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()

    alloc = _allocate_board_lot(12_300, weights)
    assert sum(alloc) == 12_300
    assert all(q >= 0 for q in alloc)


def test_capacity_cap_carryover_and_unfilled_logged() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "open": [10000, 10000, 10000, 10000, 10000],
            "high": [10050, 10050, 10050, 10050, 10050],
            "low": [9950, 9950, 9950, 9950, 9950],
            "close": [10000, 10000, 10000, 10000, 10000],
            # cap qty/day = floor(adtv*5%/open/100)*100 = 100
            "value_vnd": [20_000_000] * 5,
            "atr14": [100] * 5,
            "ceiling_price": [10700] * 5,
            "floor_price": [9300] * 5,
        }
    )
    signal = pd.Series([1, 1, 1, 1, 1], index=bars.index)

    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=1_000_000_000.0),
    )

    sched = out["execution_schedules"]
    assert len(sched) == 3
    assert (sched["filled_qty"] <= 100).all()
    assert (sched["carry_out_qty"] > 0).any()
    assert int(sched["unfilled_after_day3"].iloc[-1]) > 0


def test_reconciliation_invariant_within_1_vnd_and_qty_exact() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=7, freq="D"),
            "open": [10000, 10100, 10200, 10150, 10250, 10300, 10200],
            "high": [10100, 10200, 10300, 10250, 10300, 10400, 10300],
            "low": [9900, 10000, 10100, 10050, 10150, 10200, 10100],
            "close": [10050, 10150, 10250, 10200, 10280, 10350, 10250],
            "value_vnd": [8e9] * 7,
            "atr14": [120] * 7,
            "ceiling_price": [10700] * 7,
            "floor_price": [9300] * 7,
        }
    )
    signal = pd.Series([1, 1, 1, 0, 0, 0, 0], index=bars.index)

    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=200_000_000.0),
    )

    eq = out["equity_curve"]
    fills = out["realized_fills"]
    final = eq.iloc[-1]

    buy_flows = float(fills.loc[fills["side"] == "BUY", "gross_exec_notional"].sum())
    sell_flows = float(fills.loc[fills["side"] == "SELL", "gross_exec_notional"].sum())
    costs = float((fills["commission"] + fills["sell_tax"]).sum())
    expected_cash = 200_000_000.0 - buy_flows + sell_flows - costs
    assert abs(float(final["cash"]) - expected_cash) <= 1.0

    buy_qty = int(fills.loc[fills["side"] == "BUY", "filled_qty"].sum())
    sell_qty = int(fills.loc[fills["side"] == "SELL", "filled_qty"].sum())
    assert int(final["position_qty"]) == buy_qty - sell_qty


def test_sell_clamp_regression_still_holds_with_slices() -> None:
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
    assert (sells["filled_qty"] <= sells["order_qty"]).all()


def test_unfilled_is_logged_when_backtest_ends_before_day3() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="D"),
            "open": [10000, 10000],
            "high": [10100, 10100],
            "low": [9900, 9900],
            "close": [10000, 10000],
            "value_vnd": [20_000_000, 20_000_000],
            "atr14": [100, 100],
            "ceiling_price": [10700, 10700],
            "floor_price": [9300, 9300],
        }
    )
    signal = pd.Series([1, 1], index=bars.index)

    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=1_000_000_000.0),
    )
    sched = out["execution_schedules"]
    assert len(sched) == 1
    assert int(sched["unfilled_after_day3"].iloc[-1]) > 0

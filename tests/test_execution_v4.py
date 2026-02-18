from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.execution_v4 import (
    ExecutionV4Assumptions,
    compute_session_vwap,
    simulate_execution_v4,
)


def _toy_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2025-01-02 09:05:00", "open": 9.99, "close": 10.00, "volume": 3000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 09:20:00", "open": 10.05, "close": 10.10, "volume": 2000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 10:20:00", "open": 10.15, "close": 10.166666666666666, "volume": 3000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 13:10:00", "open": 10.15, "close": 10.15, "volume": 4000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 14:00:00", "open": 10.25, "close": 10.25, "volume": 3000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 14:35:00", "open": 10.30, "close": 10.30, "volume": 1000, "spread_bps": 1.0},
            {"timestamp": "2025-01-02 15:00:00", "open": 10.40, "close": 10.40, "volume": 500, "spread_bps": 1.0},
        ]
    )


def test_execution_v4_session_vwap_matches_golden() -> None:
    bars = _toy_bars()
    out = compute_session_vwap(bars)

    golden = json.loads(Path("tests/golden/execution_v4_session_vwap_toy.json").read_text(encoding="utf-8"))
    got_rows = out.to_dict(orient="records")

    assert len(got_rows) == len(golden["rows"])
    for got, exp in zip(got_rows, golden["rows"]):
        assert got["session"] == exp["session"]
        assert abs(float(got["session_vwap"]) - float(exp["session_vwap"])) < 1e-9
        assert abs(float(got["session_volume"]) - float(exp["session_volume"])) < 1e-9
        assert int(got["n_bars"]) == int(exp["n_bars"])


def test_execution_v4_fill_ratios_and_partial_fills() -> None:
    bars = _toy_bars()
    assumptions = ExecutionV4Assumptions(participation_limit=0.05)
    out = simulate_execution_v4(bars, symbol="AAA", side="BUY", order_qty=1000, assumptions=assumptions)
    logs = out["tca_log"]

    # open auction bar: 3000*0.05=150 cap, *0.35=52.5 => 0 after lot rounding
    row_open = logs[logs["session"] == "open_auction"].iloc[0]
    assert int(row_open["filled_qty"]) == 0

    # first continuous bar: 2000*0.05=100 cap, *0.9=90 => 0 after lot rounding
    row_am = logs[logs["timestamp"] == pd.Timestamp("2025-01-02 09:20:00")].iloc[0]
    assert int(row_am["filled_qty"]) == 0

    # second continuous bar: 3000*0.05=150 cap, *0.9=135 => 100
    row_am2 = logs[logs["timestamp"] == pd.Timestamp("2025-01-02 10:20:00")].iloc[0]
    assert int(row_am2["filled_qty"]) == 100


def test_execution_v4_reconciliation_invariants_and_logs() -> None:
    bars = _toy_bars()
    assumptions = ExecutionV4Assumptions(participation_limit=0.20, carryover_limit_ratio=0.30)
    out = simulate_execution_v4(bars, symbol="AAA", side="BUY", order_qty=1500, assumptions=assumptions)

    summary = out["summary"]
    fills = out["fills"]
    logs = out["tca_log"]

    assert summary["recon_qty_ok"] is True
    assert summary["order_qty"] == summary["executed_qty"] + summary["carryover_qty"] + summary["cancelled_qty"]
    assert summary["carryover_qty"] <= summary["carryover_limit_qty"]

    required_cols = {
        "session",
        "fill_ratio_rule",
        "base_slippage_bps",
        "session_slippage_addon_bps",
        "spread_addon_bps",
        "total_slippage_bps",
        "exec_price",
        "exec_notional",
        "remaining_before",
        "remaining_after",
    }
    assert required_cols.issubset(set(logs.columns))

    if not fills.empty:
        assert abs(float(fills["exec_notional"].sum()) - float(summary["exec_notional_total"])) < 1e-8

from __future__ import annotations

from core.portfolio.dashboard import apply_scenario_shocks


def test_scenario_shocks_apply_correctly() -> None:
    dash = {
        "nav": 1_000_000.0,
        "holdings": [
            {"symbol": "AAA", "value": 600_000.0},
            {"symbol": "BBB", "value": 200_000.0},
        ],
        "exposures": {"sector": {"Tech": 0.6, "Bank": 0.2}},
    }
    preview = {"expected_costs": {"total": 10_000.0}, "trades": [{"notional": 100_000.0}]}
    out = apply_scenario_shocks(dash, preview)
    assert out["S1_cost_x2"]["delta_expected_cost"] == 10_000.0
    assert out["S2_fill_x0_5"]["unfilled_notional"] == 50_000.0
    assert out["S3_market_down_5"]["delta_nav"] == -40_000.0
    assert out["S4_largest_sector_down_8"]["largest_sector"] == "Tech"

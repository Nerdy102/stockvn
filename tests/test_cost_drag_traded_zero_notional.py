import math

import pandas as pd

from scripts.run_eval_lab import _metrics_from_eq


def test_cost_drag_traded_zero_notional_is_nan() -> None:
    eq = pd.DataFrame(
        {
            "equity_gross": [1.0, 1.0],
            "equity_net": [1.0, 1.0],
            "turnover_l1": [0.0, 0.0],
            "commission_cost": [0.0, 0.0],
            "sell_tax_cost": [0.0, 0.0],
            "slippage_cost": [0.0, 0.0],
            "cum_cost": [0.0, 0.0],
        }
    )
    perf, _, _ = _metrics_from_eq(eq)
    assert math.isnan(perf["cost_drag_vs_traded"])

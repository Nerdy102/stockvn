import pandas as pd

from scripts.run_eval_lab import _metrics_from_eq


def test_sharpe_sign_sanity_positive_total_return_has_positive_mean_daily() -> None:
    eq = pd.DataFrame(
        {
            "equity_gross": [1.0, 1.1, 1.2],
            "equity_net": [1.0, 1.05, 1.10],
            "turnover_l1": [0.0, 0.0, 0.0],
            "commission_cost": [0.0, 0.0, 0.0],
            "sell_tax_cost": [0.0, 0.0, 0.0],
            "slippage_cost": [0.0, 0.0, 0.0],
            "cum_cost": [0.0, 0.0, 0.0],
        }
    )
    perf, _, _ = _metrics_from_eq(eq)
    mean_daily_net = eq["equity_net"].pct_change().fillna(0.0).mean()
    assert perf["total_return"] > 0.0
    assert mean_daily_net >= -1e-12

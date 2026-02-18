import numpy as np
import pandas as pd

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.market_rules import load_market_rules


def test_cash_reconciliation_invariant() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="D"),
            "open": [10000, 10200, 10100, 9900, 9800, 9700],
            "high": [10100, 10300, 10200, 10000, 9900, 9800],
            "low": [9900, 10100, 10000, 9800, 9700, 9600],
            "close": [10050, 10150, 10050, 9850, 9750, 9650],
            "value_vnd": [8e9] * 6,
            "atr14": [150] * 6,
            "ceiling_price": [10700] * 6,
            "floor_price": [9300] * 6,
        }
    )
    signal = pd.Series([1, 1, 0, 0, 1, 0], index=bars.index)
    out = run_backtest_v3(
        bars,
        signal,
        load_market_rules("configs/market_rules_vn.yaml"),
        BacktestV3Config(initial_cash=200_000_000.0),
    )
    eq = out["equity_curve"]
    assert np.allclose(eq["cash"] + eq["market_value"], eq["equity"], atol=1e-6)
    assert (eq["position_qty"] >= 0).all()
    assert (eq["cash_recon_gap"].abs() <= 1.0).all()

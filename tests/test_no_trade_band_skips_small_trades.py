from __future__ import annotations

import pandas as pd

from core.ml.portfolio_v2 import apply_no_trade_band


def test_no_trade_band_skips_small_trades() -> None:
    df = pd.DataFrame(
        [
            {"w": 0.101, "current_w": 0.100, "target_qty": 100, "order_notional": 6_000_000},
            {"w": 0.200, "current_w": 0.100, "target_qty": 90, "order_notional": 6_000_000},
            {"w": 0.200, "current_w": 0.100, "target_qty": 200, "order_notional": 4_000_000},
            {"w": 0.200, "current_w": 0.100, "target_qty": 200, "order_notional": 6_000_000},
        ]
    )
    out = apply_no_trade_band(df)
    assert len(out) == 1

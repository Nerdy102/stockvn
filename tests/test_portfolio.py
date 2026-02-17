from __future__ import annotations

import pandas as pd

from core.fees_taxes import FeesTaxes
from core.portfolio.analytics import compute_positions_avg_cost


def test_sell_clamp_fee_tax() -> None:
    fees = FeesTaxes(sell_tax_rate=0.001, dividend_tax_rate=0.05, default_commission_rate=0.0015, broker_commission={})
    trades = pd.DataFrame(
        [
            {"trade_date": "2025-01-01", "symbol": "AAA", "side": "BUY", "quantity": 100, "price": 10000, "strategy_tag": "x"},
            {"trade_date": "2025-01-02", "symbol": "AAA", "side": "SELL", "quantity": 200, "price": 11000, "strategy_tag": "x"},
        ]
    )
    latest = {"AAA": 11000}
    pos, realized = compute_positions_avg_cost(trades, latest, fees, broker_name="demo_broker")
    assert realized.shape[0] == 1
    row = realized.iloc[0].to_dict()
    assert float(row["quantity"]) == 100.0  # clamped
    notional = 100.0 * 11000.0
    assert abs(float(row["commission"]) - fees.commission(notional, "demo_broker")) < 1e-6
    assert abs(float(row["sell_tax"]) - fees.sell_tax(notional)) < 1e-6

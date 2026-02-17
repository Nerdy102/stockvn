import pandas as pd
from core.fees_taxes import FeesTaxes
from core.portfolio.analytics import compute_positions_avg_cost


def test_sell_clamp_happens_before_fee_tax() -> None:
    fees = FeesTaxes(0.001, 0.05, 0.0015, {})
    trades = pd.DataFrame(
        [
            {
                "trade_date": "2025-01-01",
                "symbol": "AAA",
                "side": "BUY",
                "quantity": 100,
                "price": 10000,
                "strategy_tag": "x",
            },
            {
                "trade_date": "2025-01-02",
                "symbol": "AAA",
                "side": "SELL",
                "quantity": 150,
                "price": 11000,
                "strategy_tag": "x",
            },
        ]
    )
    _, realized = compute_positions_avg_cost(trades, {"AAA": 11000}, fees)
    row = realized.iloc[0]
    assert row["quantity"] == 100
    assert row["commission"] == fees.commission(100 * 11000)
    assert row["sell_tax"] == fees.sell_tax(100 * 11000)

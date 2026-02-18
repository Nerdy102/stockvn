from __future__ import annotations

import numpy as np

from core.alpha_v3.portfolio import construct_portfolio_v3


def test_construct_portfolio_v3_recomputes_cash_after_no_trade_filter() -> None:
    rng = np.random.default_rng(123)
    n = 4
    symbols = [f"S{i}" for i in range(n)]
    returns = rng.normal(0.0, 0.01, size=(252, n))

    # Full-risk current holdings (sum=1). If all trades are skipped by no-trade band,
    # final weights should remain here and cash must be recomputed to 0.
    current_w = np.array([0.25, 0.25, 0.25, 0.25])
    nav = 2_000_000_000.0
    next_open = np.array([50_000.0, 50_000.0, 50_000.0, 50_000.0])
    adtv = np.array([400e9, 400e9, 400e9, 400e9])
    atr14 = np.array([400.0, 400.0, 400.0, 400.0])
    close = next_open.copy()
    spread = np.array([0.001, 0.001, 0.001, 0.001])
    sectors = ["A", "B", "C", "D"]

    final_w, cash_w, intents = construct_portfolio_v3(
        symbols=symbols,
        returns_252=returns,
        current_w=current_w,
        nav=nav,
        next_open_prices=next_open,
        adtv=adtv,
        atr14=atr14,
        close=close,
        spread_proxy=spread,
        sectors=sectors,
        band=1.0,  # force no-trade for all names
    )

    assert np.allclose(final_w, current_w)
    assert np.isclose(cash_w, 0.0)
    assert intents == []

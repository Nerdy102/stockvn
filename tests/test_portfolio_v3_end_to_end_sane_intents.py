from __future__ import annotations

import numpy as np

from core.alpha_v3.portfolio import construct_portfolio_v3


def test_portfolio_v3_end_to_end_sane_intents() -> None:
    rng = np.random.default_rng(7)
    n = 6
    symbols = [f"S{i}" for i in range(n)]
    returns = rng.normal(0.0, 0.01, size=(252, n))
    current_w = np.full(n, 1.0 / n)
    nav = 10_000_000_000.0
    next_open = np.array([20_000.0, 30_000.0, 40_000.0, 25_000.0, 50_000.0, 35_000.0])
    adtv = np.array([500e9, 300e9, 200e9, 150e9, 350e9, 250e9])
    atr14 = np.array([500, 700, 800, 450, 900, 650], dtype=float)
    close = next_open.copy()
    spread = np.array([0.0005, 0.0008, 0.0007, 0.0010, 0.0006, 0.0009])
    sectors = ["FIN", "FIN", "TECH", "TECH", "IND", "UTIL"]

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
    )

    assert np.all(final_w >= 0)
    assert np.isclose(final_w.sum() + cash_w, 1.0)
    assert cash_w >= 0.10 - 1e-12
    assert np.abs(final_w - current_w).sum() / 2.0 <= 0.30 + 1e-12

    for intent in intents:
        assert intent["side"] in {"BUY", "SELL"}
        assert intent["qty"] % 100 == 0
        assert intent["qty"] >= 100
        assert intent["ref_price"] > 0
        assert intent["target_weight"] >= 0
        assert isinstance(intent["reason"], str) and intent["reason"]

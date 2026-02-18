from __future__ import annotations

import numpy as np
import pytest

from core.alpha_v3.portfolio import construct_portfolio_v3, generate_trade_intents


def test_construct_portfolio_v3_raises_on_length_mismatch() -> None:
    returns = np.random.default_rng(1).normal(0.0, 0.01, size=(252, 3))
    with pytest.raises(ValueError, match="symbols length must match number of columns"):
        construct_portfolio_v3(
            symbols=["A", "B"],
            returns_252=returns,
            current_w=np.array([0.3, 0.3, 0.4]),
            nav=1_000_000_000.0,
            next_open_prices=np.array([20_000.0, 30_000.0, 25_000.0]),
            adtv=np.array([1e9, 1e9, 1e9]),
            atr14=np.array([500.0, 600.0, 550.0]),
            close=np.array([20_000.0, 30_000.0, 25_000.0]),
            spread_proxy=np.array([0.001, 0.001, 0.001]),
            sectors=["A", "B", "C"],
        )


def test_generate_trade_intents_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="symbols length must match target weights"):
        generate_trade_intents(
            symbols=["A"],
            target_w=np.array([0.6, 0.4]),
            current_w=np.array([0.5, 0.5]),
            nav=1_000_000_000.0,
            next_open_prices=np.array([20_000.0, 30_000.0]),
        )

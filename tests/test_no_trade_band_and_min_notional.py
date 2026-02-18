from __future__ import annotations

import numpy as np

from core.alpha_v3.portfolio import apply_no_trade_band


def test_no_trade_band_and_min_notional() -> None:
    current = np.array([0.10, 0.10, 0.10, 0.10])
    target = np.array([0.101, 0.12, 0.15, 0.16])
    prices = np.array([50_000.0, 50_000.0, 50_000.0, 50_000.0])
    nav = 1_000_000_000.0

    # idx0 blocked by no-trade band
    # idx1 blocked by qty < 100 after floor
    # idx2 blocked by notional < 5m (qty 100 -> 5m passes, so set lower)
    # idx3 should pass
    target[1] = current[1] + 0.004  # > band but qty=80
    target[2] = current[2] + 0.005  # qty=100, notional=5m exactly passes
    prices[2] = 40_000.0  # now notional=4m fails

    out = apply_no_trade_band(target, current, prices, nav)

    assert np.isclose(out[0], current[0])
    assert np.isclose(out[1], current[1])
    assert np.isclose(out[2], current[2])
    assert np.isclose(out[3], target[3])

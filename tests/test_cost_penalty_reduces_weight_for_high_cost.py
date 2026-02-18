from __future__ import annotations

import numpy as np

from core.alpha_v3.costs import apply_cost_penalty_to_weights


def test_cost_penalty_reduces_weight_for_high_cost() -> None:
    w = np.array([0.5, 0.5])
    target_notional = np.array([10_000_000.0, 10_000_000.0])
    adtv = np.array([2_000_000_000.0, 50_000_000.0])
    atr14 = np.array([1.0, 10.0])
    close = np.array([100.0, 20.0])
    spread = np.array([0.0005, 0.005])

    out = apply_cost_penalty_to_weights(w, target_notional, adtv, atr14, close, spread)
    assert np.isclose(out.sum(), 1.0)
    assert out[1] < out[0]

from __future__ import annotations

import numpy as np

from core.alpha_v3.hrp import compute_hrp_weights


def test_hrp_nonneg_sum1_and_order_invariance() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=(252, 8))

    w = compute_hrp_weights(returns)
    assert np.all(w >= -1e-12)
    assert np.isclose(w.sum(), 1.0)

    perm = np.array([3, 0, 6, 1, 7, 4, 2, 5])
    w_perm = compute_hrp_weights(returns[:, perm])
    restored = np.zeros_like(w_perm)
    restored[perm] = w_perm
    assert np.allclose(w, restored, atol=1e-6)

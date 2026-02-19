from __future__ import annotations

import numpy as np

from core.portfolio_alloc.hrp import hrp_weights


def test_hrp_weights_valid() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.01, size=(252, 10))
    w = hrp_weights(x)
    assert w.shape[0] == 10
    assert abs(float(np.sum(w)) - 1.0) < 1e-9
    assert float(np.min(w)) >= 0.0

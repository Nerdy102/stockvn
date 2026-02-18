from __future__ import annotations

import numpy as np

from core.alpha_v3.hrp import compute_hrp_weights


def test_hrp_handles_singular_cov_with_ledoitwolf() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=(252, 1))
    # Perfectly collinear matrix => singular sample covariance
    returns = np.hstack([base, base, base * 2.0, base * -1.0])

    w = compute_hrp_weights(returns)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)

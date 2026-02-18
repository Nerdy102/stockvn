from __future__ import annotations

import numpy as np
import pandas as pd

from core.alpha_v3.bootstrap import block_bootstrap_ci


def test_bootstrap_ci_has_bounds_and_order() -> None:
    r = pd.Series(np.random.default_rng(10).normal(0.0007, 0.01, size=350))
    ci = block_bootstrap_ci(r, block=20, n_resamples=400)
    assert ci["sharpe_lo"] <= ci["sharpe_hi"]
    assert ci["cagr_lo"] <= ci["cagr_hi"]

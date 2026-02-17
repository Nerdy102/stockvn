from __future__ import annotations

import numpy as np
import pandas as pd

from core.ml.diagnostics import block_bootstrap_ci


def test_block_bootstrap_ci_shapes() -> None:
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, size=300))
    ci = block_bootstrap_ci(r, block=20, n_resamples=200)
    assert set(ci.keys()) == {"sharpe_lo", "sharpe_hi", "cagr_lo", "cagr_hi"}
    assert all(np.isfinite(list(ci.values())))

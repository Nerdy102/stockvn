from __future__ import annotations

import numpy as np
import pandas as pd

from core.alpha_v3.dsr import compute_deflated_sharpe_ratio


def test_dsr_formula_known_case() -> None:
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0012, 0.01, size=252))
    out = compute_deflated_sharpe_ratio(r, n_trials=36)
    assert 0.0 <= out.dsr_value <= 1.0
    assert out.components["n_trials"] == 36.0
    assert "z" in out.components
    assert np.isfinite(out.components["sr0_hat"])

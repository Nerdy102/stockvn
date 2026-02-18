from __future__ import annotations

import numpy as np
import pandas as pd

from core.alpha_v3.pbo import compute_pbo_cscv


def test_pbo_phi_small_when_true_variant_best_oos() -> None:
    rng = np.random.default_rng(77)
    base = rng.normal(0.0, 0.01, size=(300, 8))
    data = pd.DataFrame(base, columns=[f"v{i}" for i in range(8)])
    data["v0"] = rng.normal(0.0012, 0.01, size=300)
    phi, _ = compute_pbo_cscv(data, slices=10)
    assert phi < 0.35

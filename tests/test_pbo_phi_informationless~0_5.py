from __future__ import annotations

import numpy as np
import pandas as pd

from core.alpha_v3.pbo import compute_pbo_cscv


def test_pbo_phi_informationless_close_to_half() -> None:
    rng = np.random.default_rng(123)
    data = pd.DataFrame(rng.normal(0.0, 0.01, size=(300, 8)), columns=[f"v{i}" for i in range(8)])
    phi, summary = compute_pbo_cscv(data, slices=10)
    assert abs(phi - 0.5) < 0.25
    assert summary["n_logits"] > 0

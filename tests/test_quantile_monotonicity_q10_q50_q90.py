from __future__ import annotations

import numpy as np

from core.alpha_v3.models import AlphaV3ModelBundle


def test_quantile_monotonicity_q10_q50_q90() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 6))
    y = 0.3 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.15, size=500)
    model = AlphaV3ModelBundle().fit(x, y)
    comp = model.predict_components(x[:100])

    assert np.all(comp["hgbr_q10_v3"] <= comp["hgbr_q50_v3"])
    assert np.all(comp["hgbr_q50_v3"] <= comp["hgbr_q90_v3"])

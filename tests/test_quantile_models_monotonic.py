from __future__ import annotations

import numpy as np

from core.ml.models_v2 import MlModelV2Bundle


def test_quantile_models_monotonic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 4))
    y = x[:, 0] * 0.1 + rng.normal(scale=0.2, size=600)
    m = MlModelV2Bundle().fit(x, y)
    comp = m.predict_components(x[:50])
    assert np.all(comp["hgbr_q10_v2"] <= comp["hgbr_q50_v2"])
    assert np.all(comp["hgbr_q50_v2"] <= comp["hgbr_q90_v2"])

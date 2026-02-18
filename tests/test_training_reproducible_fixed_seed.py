from __future__ import annotations

import numpy as np

from core.alpha_v3.models import AlphaV3ModelBundle


def test_training_reproducible_fixed_seed() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 8))
    y = x[:, 0] * 0.4 - x[:, 3] * 0.2 + rng.normal(scale=0.1, size=400)

    m1 = AlphaV3ModelBundle().fit(x, y)
    m2 = AlphaV3ModelBundle().fit(x, y)

    p1 = m1.predict_components(x[:120])["score"]
    p2 = m2.predict_components(x[:120])["score"]
    assert np.allclose(p1, p2, atol=1e-12)

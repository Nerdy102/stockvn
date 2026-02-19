from __future__ import annotations

import numpy as np

from core.quant_validation_advanced.cscv_pbo import compute_cscv_pbo


def test_cscv_pbo_sanity() -> None:
    t = 240
    good = np.full((t, 1), 0.01)
    bad = np.full((t, 5), -0.001)
    m1 = np.hstack([good, bad])
    rep1, _ = compute_cscv_pbo(m1, s_segments=8)
    assert rep1 is not None
    assert rep1.pbo_phi <= 0.05

    m2 = np.full((t, 6), 0.001)
    rep2, _ = compute_cscv_pbo(m2, s_segments=8)
    assert rep2 is not None
    assert abs(rep2.logits_p50) < 1e-9
    assert 0.3 <= rep2.pbo_phi <= 0.7

from __future__ import annotations

import numpy as np

from core.quant_validation_advanced.cscv_pbo import compute_cscv_pbo


def test_cscv_pbo_deterministic() -> None:
    rng = np.random.default_rng(42)
    m = rng.normal(0.001, 0.01, size=(240, 6))
    r1, _ = compute_cscv_pbo(m, s_segments=8)
    r2, _ = compute_cscv_pbo(m, s_segments=8)
    assert r1 is not None and r2 is not None
    assert r1.pbo_phi == r2.pbo_phi
    assert r1.logits_p50 == r2.logits_p50

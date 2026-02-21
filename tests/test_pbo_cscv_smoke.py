import numpy as np

from core.eval_lab.multiple_testing import pbo_cscv


def test_pbo_cscv_smoke() -> None:
    rng = np.random.default_rng(42)
    mat = rng.normal(0.0, 0.01, size=(300, 5))
    v = pbo_cscv(mat, s=16, seed=42)
    assert 0.0 <= v <= 1.0

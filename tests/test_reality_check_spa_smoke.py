import numpy as np

from core.eval_lab.multiple_testing import reality_check, spa


def test_reality_check_spa_smoke() -> None:
    rng = np.random.default_rng(42)
    diff = rng.normal(0.0, 0.01, size=(200, 4))
    rc = reality_check(diff, block_size=10, b=100, seed=42)
    sp = spa(diff, block_size=10, b=100, seed=42)
    assert 0.0 <= rc <= 1.0
    assert 0.0 <= sp <= 1.0

import numpy as np

from core.eval_lab.multiple_testing import format_pvalue, reality_check, spa


def test_pvalue_formatting_bounds() -> None:
    rng = np.random.default_rng(42)
    diff = rng.normal(0.0, 0.01, size=(120, 3))
    b = 200
    p1 = reality_check(diff, block_size=10, b=b, seed=42)
    p2 = spa(diff, block_size=10, b=b, seed=42)
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0
    assert p1 >= 1.0 / b
    assert p2 >= 1.0 / b
    s = format_pvalue(p1, b)
    assert s.startswith("p =") or s.startswith("p <")

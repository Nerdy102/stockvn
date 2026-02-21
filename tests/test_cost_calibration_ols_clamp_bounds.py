from core.costs.calibration import _winsorize


def test_winsorize_deterministic_and_bounds() -> None:
    import numpy as np

    x = np.array([1, 2, 3, 100], dtype=float)
    y1 = _winsorize(x, 1, 99)
    y2 = _winsorize(x, 1, 99)
    assert y1.tolist() == y2.tolist()
    assert y1.max() <= 100

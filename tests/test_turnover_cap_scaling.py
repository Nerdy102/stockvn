from __future__ import annotations

import numpy as np

from core.alpha_v3.portfolio import cap_turnover


def test_turnover_cap_scaling() -> None:
    current = np.array([0.25, 0.25, 0.25, 0.25])
    target = np.array([0.55, 0.15, 0.15, 0.15])

    out = cap_turnover(target, current, max_turnover=0.30)
    turnover = np.abs(out - current).sum() / 2.0
    assert turnover <= 0.30 + 1e-12

    # Unconstrained turnover would be 0.30 exactly here if one side +0.30 others -0.10 each.
    # Push stronger target to force scaling.
    target2 = np.array([0.85, 0.05, 0.05, 0.05])
    out2 = cap_turnover(target2, current, max_turnover=0.30)
    turnover2 = np.abs(out2 - current).sum() / 2.0
    assert np.isclose(turnover2, 0.30)

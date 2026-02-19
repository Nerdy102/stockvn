from __future__ import annotations

import numpy as np

from core.quant_stats.psr_dsr import probabilistic_sharpe_ratio


def test_quant_psr_formula_bounds_and_monotonic() -> None:
    t = 252
    g3 = 0.0
    g4 = 3.0
    psr_low = probabilistic_sharpe_ratio(0.05, 0.0, t, g3, g4)
    psr_high = probabilistic_sharpe_ratio(0.25, 0.0, t, g3, g4)
    assert 0.0 <= psr_low <= 1.0
    assert 0.0 <= psr_high <= 1.0
    assert psr_high >= psr_low

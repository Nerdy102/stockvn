from __future__ import annotations

import numpy as np
import pandas as pd

from research.stats.psr_mintrl import compute_mintrl, compute_psr


def test_psr_mintrl_sanity_on_synthetic() -> None:
    rng = np.random.default_rng(123)
    good = pd.Series(rng.normal(loc=0.0015, scale=0.01, size=800))
    bad = pd.Series(rng.normal(loc=0.0, scale=0.01, size=800))

    psr_good = compute_psr(good, sr_threshold=0.0)
    psr_bad = compute_psr(bad, sr_threshold=0.0)

    assert 0.0 <= psr_good.psr_value <= 1.0
    assert 0.0 <= psr_bad.psr_value <= 1.0
    assert psr_good.psr_value > 0.95
    assert psr_bad.psr_value < psr_good.psr_value

    mintrl_good = compute_mintrl(good, sr_threshold=0.0, alpha=0.95)
    assert mintrl_good.mintrl > 0
    assert mintrl_good.mintrl < len(good)

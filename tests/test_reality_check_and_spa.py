from __future__ import annotations

import numpy as np
import pandas as pd

from research.stats.reality_check import white_reality_check
from research.stats.spa_test import hansen_spa_test


def test_reality_check_informationless_competitors_approximately_uniform() -> None:
    rng = np.random.default_rng(7)
    pvals: list[float] = []
    for i in range(24):
        bench = pd.Series(rng.normal(0.0, 0.01, size=260))
        comp = pd.DataFrame(
            {
                f"v{j}": pd.Series(rng.normal(0.0, 0.01, size=260))
                for j in range(6)
            }
        )
        p, _ = white_reality_check(bench, comp, n_bootstrap=300, block_mean=20.0, seed=100 + i)
        pvals.append(p)

    mean_p = float(np.mean(pvals))
    assert 0.35 <= mean_p <= 0.65


def test_spa_more_powerful_or_equal_than_rc_on_constructed_fixture() -> None:
    rng = np.random.default_rng(1234)
    t = 400
    bench = pd.Series(rng.normal(0.0, 0.01, size=t))
    comp = pd.DataFrame(
        {
            "weak": pd.Series(rng.normal(0.0001, 0.01, size=t)),
            "strong": pd.Series(rng.normal(0.0010, 0.01, size=t)),
            "noise": pd.Series(rng.normal(0.0, 0.01, size=t)),
        }
    )

    rc_p, _ = white_reality_check(bench, comp, n_bootstrap=600, block_mean=20.0, seed=99)
    spa_p, _ = hansen_spa_test(bench, comp, n_bootstrap=600, block_mean=20.0, seed=99)

    assert spa_p <= rc_p

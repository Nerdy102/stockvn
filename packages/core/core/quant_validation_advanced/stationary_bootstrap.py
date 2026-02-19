from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(n: int, q: float = 0.9, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.zeros(n, dtype=int)
    idx[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if float(rng.random()) <= q:
            idx[t] = (idx[t - 1] + 1) % n
        else:
            idx[t] = int(rng.integers(0, n))
    return idx

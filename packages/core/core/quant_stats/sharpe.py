from __future__ import annotations

import math
import numpy as np


def sharpe_non_annualized(returns, rf: float = 0.0) -> float:
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    ex = arr - float(rf)
    std = float(np.std(ex, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(ex) / std)


def annualize_sharpe(sr: float, periods_per_year: int) -> float:
    return float(sr) * math.sqrt(float(periods_per_year))

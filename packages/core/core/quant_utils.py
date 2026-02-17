from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clip outliers by quantile bounds (cross-sectional safe)."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = x.dropna()
    if valid.empty:
        return x
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    if lo > hi:
        lo, hi = hi, lo
    return x.clip(lower=lo, upper=hi)


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index, dtype=float)
    return (x - mu) / sd


def robust_zscore(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    return zscore_series(winsorize_series(s, lower_q=lower_q, upper_q=upper_q))


def neutralize_by_group(score: pd.Series, group: pd.Series) -> pd.Series:
    """Industry/size bucket neutralization by subtracting group means."""
    s = pd.to_numeric(score, errors="coerce").astype(float)
    g = group.astype(str)
    out = s.copy()
    for _, idx in g.groupby(g).groups.items():
        grp = s.loc[idx]
        out.loc[idx] = grp - float(grp.mean())
    return out

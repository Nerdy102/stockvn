from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(x: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    if x.empty:
        return x
    lo, hi = x.quantile(p_low), x.quantile(p_high)
    return x.clip(lower=lo, upper=hi)


def zscore_cross_section(
    df: pd.DataFrame, by_date_col: str = "date", value_col: str = "value"
) -> pd.Series:
    def _z(g: pd.DataFrame) -> pd.Series:
        s = g[value_col].astype(float)
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=g.index)
        return (s - s.mean()) / sd

    return df.groupby(by_date_col, group_keys=False).apply(_z)


def neutralize_by_group(z: pd.Series, group: pd.Series) -> pd.Series:
    g = pd.DataFrame({"z": z, "group": group})
    return g["z"] - g.groupby("group")["z"].transform("mean")


def neutralize_by_size(z: pd.Series, size_metric: pd.Series, n_bins: int = 10) -> pd.Series:
    df = pd.DataFrame({"z": z, "size": size_metric}).dropna()
    if df.empty:
        return z * 0.0
    df["bin"] = pd.qcut(df["size"], q=min(n_bins, max(2, df["size"].nunique())), duplicates="drop")
    out = df["z"] - df.groupby("bin")["z"].transform("mean")
    return out.reindex(z.index).fillna(0.0)

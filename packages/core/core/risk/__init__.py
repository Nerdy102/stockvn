from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(price: pd.Series) -> float:
    p = price.dropna()
    if p.empty:
        return 0.0
    peak = p.cummax()
    dd = (p / peak) - 1.0
    return float(dd.min())


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(np.quantile(r, alpha))


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if joined.shape[0] < 10:
        return 0.0
    joined.columns = ["asset", "bench"]
    var_b = float(np.var(joined["bench"].values, ddof=1))
    if var_b == 0.0:
        return 0.0
    cov = float(np.cov(joined["bench"].values, joined["asset"].values, ddof=1)[0, 1])
    return cov / var_b


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class PsrResult:
    psr_value: float
    sr_hat: float
    sr_threshold: float
    t: int
    skew: float
    kurt: float


@dataclass(frozen=True)
class MinTrlResult:
    mintrl: int
    sr_hat: float
    sr_threshold: float
    alpha: float


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _annualized_sr(r: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(r) < 2:
        return 0.0
    return float((r.mean() / (r.std(ddof=0) + 1e-12)) * math.sqrt(periods_per_year))


def compute_psr(
    returns: pd.Series,
    sr_threshold: float = 0.0,
    periods_per_year: float = 252.0,
) -> PsrResult:
    """
    Probabilistic Sharpe Ratio: P(SR_true > SR_threshold).

    Uses finite-sample correction based on skew/kurtosis as in Bailey & López de Prado.
    """
    r = _clean_returns(returns)
    t = int(len(r))
    if t < 2:
        return PsrResult(psr_value=0.0, sr_hat=0.0, sr_threshold=float(sr_threshold), t=t, skew=0.0, kurt=3.0)

    sr_hat = _annualized_sr(r, periods_per_year=periods_per_year)
    skew = float(r.skew()) if t > 2 else 0.0
    kurt = float(r.kurt() + 3.0) if t > 3 else 3.0

    denom = max(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2), 1e-12)
    z = ((sr_hat - float(sr_threshold)) * math.sqrt(max(t - 1, 1))) / math.sqrt(denom)
    psr = float(norm.cdf(z))
    return PsrResult(
        psr_value=psr,
        sr_hat=float(sr_hat),
        sr_threshold=float(sr_threshold),
        t=t,
        skew=float(skew),
        kurt=float(kurt),
    )


def compute_mintrl(
    returns: pd.Series,
    sr_threshold: float = 0.0,
    alpha: float = 0.95,
    periods_per_year: float = 252.0,
) -> MinTrlResult:
    """
    Minimum track-record length needed to have confidence alpha that SR_true > SR_threshold.
    """
    r = _clean_returns(returns)
    if len(r) < 2:
        return MinTrlResult(mintrl=math.inf, sr_hat=0.0, sr_threshold=float(sr_threshold), alpha=float(alpha))

    sr_hat = _annualized_sr(r, periods_per_year=periods_per_year)
    if sr_hat <= sr_threshold:
        return MinTrlResult(mintrl=math.inf, sr_hat=float(sr_hat), sr_threshold=float(sr_threshold), alpha=float(alpha))

    skew = float(r.skew()) if len(r) > 2 else 0.0
    kurt = float(r.kurt() + 3.0) if len(r) > 3 else 3.0
    z_alpha = float(norm.ppf(alpha))

    numer = max(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2), 1e-12) * (z_alpha**2)
    denom = max((sr_hat - float(sr_threshold)) ** 2, 1e-12)
    mintrl = int(math.ceil(1.0 + numer / denom))
    return MinTrlResult(mintrl=mintrl, sr_hat=float(sr_hat), sr_threshold=float(sr_threshold), alpha=float(alpha))

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


@dataclass(frozen=True)
class PsrResult:
    sr_hat: float
    sr_threshold: float
    t: int
    skew: float
    kurt: float
    psr_value: float


@dataclass(frozen=True)
class MinTrlResult:
    mintrl: int
    sr_hat: float
    sr_threshold: float
    alpha: float


def _annualized_sharpe(returns: pd.Series) -> float:
    r = returns.astype(float).dropna()
    if r.empty:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((r.mean() / sd) * np.sqrt(252.0))


def compute_psr(returns: pd.Series, sr_threshold: float = 0.0) -> PsrResult:
    r = returns.astype(float).dropna()
    t = int(len(r))
    sr_hat = _annualized_sharpe(r)
    if t <= 2:
        return PsrResult(
            sr_hat=sr_hat, sr_threshold=float(sr_threshold), t=t, skew=0.0, kurt=3.0, psr_value=0.5
        )
    sk = float(skew(r, bias=False)) if t > 2 else 0.0
    ku = float(kurtosis(r, fisher=False, bias=False)) if t > 3 else 3.0
    z = (sr_hat - float(sr_threshold)) * np.sqrt(max(1.0, t - 1.0))
    psr_val = float(norm.cdf(z))
    return PsrResult(
        sr_hat=sr_hat, sr_threshold=float(sr_threshold), t=t, skew=sk, kurt=ku, psr_value=psr_val
    )


def compute_mintrl(
    returns: pd.Series, sr_threshold: float = 0.0, alpha: float = 0.95
) -> MinTrlResult:
    r = returns.astype(float).dropna()
    sr_hat = _annualized_sharpe(r)
    edge = max(1e-9, sr_hat - float(sr_threshold))
    z = float(norm.ppf(alpha))
    n = int(np.ceil((z / edge) ** 2 + 1.0))
    return MinTrlResult(
        mintrl=max(1, n), sr_hat=sr_hat, sr_threshold=float(sr_threshold), alpha=float(alpha)
    )

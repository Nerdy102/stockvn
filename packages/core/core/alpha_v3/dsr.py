from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class DsrResult:
    dsr_value: float
    components: dict[str, float]


def _moments(returns: pd.Series) -> tuple[float, float, float]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0, 0.0, 3.0
    sr = float((r.mean() / (r.std(ddof=0) + 1e-12)) * math.sqrt(252.0))
    skew = float(r.skew()) if len(r) > 2 else 0.0
    kurt = float(r.kurt() + 3.0) if len(r) > 3 else 3.0
    return sr, skew, kurt


def compute_deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> DsrResult:
    """
    Bailey & López de Prado style DSR.

    z = ((SR - SR0_hat) * sqrt(T-1)) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2)
    DSR = Phi(z)
    SR0_hat = sqrt(var_sr) * ((1-gamma)*z1 + gamma*z2)
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    t = int(len(r))
    if t < 2:
        return DsrResult(dsr_value=0.0, components={"z": 0.0, "sr0_hat": 0.0, "sr": 0.0, "n_trials": float(max(1, n_trials)), "skew": 0.0, "kurt": 3.0, "n_obs": float(t)})

    sr, skew, kurt = _moments(r)
    trials = max(int(n_trials), 1)
    z1 = float(norm.ppf(1.0 - (1.0 / trials)))
    z2 = float(norm.ppf(1.0 - (1.0 / (trials * math.e))))
    gamma_e = 0.5772156649015329
    var_sr = max((1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)) / max(t - 1, 1), 1e-12)
    sr0_hat = math.sqrt(var_sr) * ((1.0 - gamma_e) * z1 + gamma_e * z2)

    denom = max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2), 1e-12)
    z = ((sr - sr0_hat) * math.sqrt(max(t - 1, 1))) / math.sqrt(denom)
    dsr = float(norm.cdf(z))
    return DsrResult(
        dsr_value=dsr,
        components={
            "z": float(z),
            "sr0_hat": float(sr0_hat),
            "sr": float(sr),
            "n_trials": float(trials),
            "skew": float(skew),
            "kurt": float(kurt),
            "n_obs": float(t),
            "var_sr": float(var_sr),
            "z_1": float(z1),
            "z_2": float(z2),
        },
    )

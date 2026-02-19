from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_EULER = 0.5772156649015329
_NORM = NormalDist()


def _phi(z: float) -> float:
    return float(_NORM.cdf(z))


def _ppf(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return float(_NORM.inv_cdf(p))


def probabilistic_sharpe_ratio(sr_hat: float, sr_star: float, t: int, gamma3: float, gamma4: float) -> float:
    if t <= 1:
        return 0.0
    denom = math.sqrt(max(1e-12, 1 - gamma3 * sr_hat + ((gamma4 - 1) / 4.0) * (sr_hat**2)))
    z = (sr_hat - sr_star) * math.sqrt(t - 1) / denom
    return _phi(z)


def min_track_record_length(sr_hat: float, sr_star: float, gamma3: float, gamma4: float, alpha: float = 0.05) -> tuple[float | None, str]:
    if sr_hat <= sr_star:
        return None, "SR không vượt ngưỡng"
    z_alpha = _ppf(1 - alpha)
    base = max(1e-12, 1 - gamma3 * sr_hat + ((gamma4 - 1) / 4.0) * (sr_hat**2))
    val = 1 + base * (z_alpha / (sr_hat - sr_star)) ** 2
    return float(val), "OK"


def sr0_from_trials(sr_trial_list: list[float], n_eff: int) -> tuple[float, float]:
    arr = np.asarray(sr_trial_list, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0, 0.0
    v_sr = float(np.var(arr, ddof=1))
    sigma_sr = math.sqrt(max(v_sr, 1e-12))
    n_eff = max(int(n_eff), 2)
    sr0 = sigma_sr * ((1 - _EULER) * _ppf(1 - 1 / n_eff) + _EULER * _ppf(1 - 1 / (n_eff * math.e)))
    return float(sr0), v_sr


def deflated_sharpe_ratio(sr_hat: float, t: int, gamma3: float, gamma4: float, sr_trial_list: list[float], n_eff: int) -> tuple[float, float, float]:
    sr0, v_sr = sr0_from_trials(sr_trial_list=sr_trial_list, n_eff=n_eff)
    if t <= 1:
        return 0.0, sr0, v_sr
    denom0 = math.sqrt(max(1e-12, 1 - gamma3 * sr0 + ((gamma4 - 1) / 4.0) * (sr0**2)))
    z0 = (sr_hat - sr0) * math.sqrt(t - 1) / denom0
    return _phi(z0), sr0, v_sr

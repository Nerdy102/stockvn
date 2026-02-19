from __future__ import annotations

import math

import numpy as np

from .schemas import RcSpaReport
from .stationary_bootstrap import stationary_bootstrap_indices


def compute_rc_spa(strategy_matrix, benchmark_returns, bootstrap_b: int = 300, q: float = 0.9, seed: int = 42) -> tuple[RcSpaReport, str | None]:
    m = np.asarray(strategy_matrix, dtype=float)
    bmk = np.asarray(benchmark_returns, dtype=float)
    n, k = m.shape
    bootstrap_b = min(max(int(bootstrap_b), 50), 500)
    if k > 60:
        return RcSpaReport(benchmark_name_vi="Mua và nắm giữ (Buy & Hold)", rc_stat=0.0, rc_pvalue=None, spa_stat=0.0, spa_pvalue=None, bootstrap_b=bootstrap_b, q_param=q, notes_vi="quá nhiều trial"), "quá nhiều trial"
    if n < 200:
        d = m - bmk.reshape(-1, 1)
        rc_n = float(np.max(np.sqrt(n) * np.mean(d, axis=0)))
        spa_n = float(max(rc_n, 0.0))
        return RcSpaReport(benchmark_name_vi="Mua và nắm giữ (Buy & Hold)", rc_stat=rc_n, rc_pvalue=None, spa_stat=spa_n, spa_pvalue=None, bootstrap_b=bootstrap_b, q_param=q, notes_vi="sample quá ngắn"), "sample quá ngắn"

    d = m - bmk.reshape(-1, 1)
    mean_d = np.mean(d, axis=0)
    rc_n = float(np.max(np.sqrt(n) * mean_d))
    spa_n = float(max(rc_n, 0.0))
    d0 = d - mean_d
    sigma = np.std(d, axis=0, ddof=1)
    a_n = -sigma * math.sqrt(max(1e-12, 2.0 * math.log(max(math.log(n), 1.000001))))
    mu_hat = mean_d * ((np.sqrt(n) * mean_d) <= a_n)

    rc_star = []
    spa_star = []
    for i in range(bootstrap_b):
        idx = stationary_bootstrap_indices(n=n, q=q, seed=seed + i)
        d_star = d0[idx, :]
        mean_star = np.mean(d_star, axis=0)
        rc_star_val = float(np.max(np.sqrt(n) * mean_star))
        rc_star.append(rc_star_val)
        spa_star_val = float(max(float(np.max(np.sqrt(n) * (mean_star + mu_hat))), 0.0))
        spa_star.append(spa_star_val)

    rc_p = float(np.mean(np.asarray(rc_star) >= rc_n))
    spa_p = float(np.mean(np.asarray(spa_star) >= spa_n))
    rep = RcSpaReport(
        benchmark_name_vi="Mua và nắm giữ (Buy & Hold)",
        rc_stat=rc_n,
        rc_pvalue=rc_p,
        spa_stat=spa_n,
        spa_pvalue=spa_p,
        bootstrap_b=bootstrap_b,
        q_param=q,
        notes_vi="RC/SPA dùng stationary bootstrap; p-value chỉ mang tính tham khảo.",
    )
    return rep, None

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


_NORM = NormalDist()


def psr(sr_hat: float, sr0: float, n_obs: int) -> float:
    if n_obs <= 2:
        return 0.0
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1)
    return float(_NORM.cdf(z))


def dsr(sr_hat: float, n_obs: int, n_trials: int) -> float:
    if n_obs <= 2:
        return 0.0
    penalty = math.sqrt(2.0 * math.log(max(2, n_trials))) / math.sqrt(max(1, n_obs))
    return psr(sr_hat - penalty, sr0=0.0, n_obs=n_obs)


def min_trl(sr_hat: float, sr0: float, alpha: float = 0.05) -> int:
    if sr_hat <= sr0:
        return 10**9
    z = _NORM.inv_cdf(1 - alpha)
    t = 1 + (z / max(1e-8, sr_hat - sr0)) ** 2
    return int(math.ceil(t))


def _max_stat(diff: np.ndarray) -> float:
    m = np.mean(diff, axis=0)
    return float(np.max(np.sqrt(diff.shape[0]) * m))


def _bootstrap_pvalue(obs: float, stars: np.ndarray, b: int) -> float:
    count = int(np.sum(stars >= obs))
    p = max(1.0 / max(1, b), count / max(1, b))
    assert 0.0 <= p <= 1.0
    return float(p)


def reality_check(diff: np.ndarray, block_size: int, b: int, seed: int) -> float:
    n, _ = diff.shape
    obs = _max_stat(diff)
    rng = np.random.default_rng(seed)
    blocks = max(1, math.ceil(n / block_size))
    star = []
    for _ in range(b):
        idx = []
        for _ in range(blocks):
            s = int(rng.integers(0, max(1, n - block_size + 1)))
            idx.extend(list(range(s, min(n, s + block_size))))
        boot = diff[np.asarray(idx[:n], dtype=int), :]
        boot = boot - np.mean(boot, axis=0, keepdims=True)
        star.append(_max_stat(boot))
    p = _bootstrap_pvalue(obs=obs, stars=np.asarray(star, dtype=float), b=b)
    return p


def spa(diff: np.ndarray, block_size: int, b: int, seed: int) -> float:
    return reality_check(diff=diff, block_size=block_size, b=b, seed=seed + 17)


def format_pvalue(p: float, b: int) -> str:
    if p < (1.0 / max(1, b)) + 1e-15:
        return f"p < {1/max(1,b):.6f}"
    return f"p = {p:.6f}"


def benjamini_hochberg(pvals: dict[str, float], q: float) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k in pvals}
    cutoff = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * q:
            cutoff = i
    for i, (k, _) in enumerate(items, start=1):
        out[k] = i <= cutoff
    return out


def pbo_cscv(ret_matrix: np.ndarray, s: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t, m = ret_matrix.shape
    probs = []
    for _ in range(s):
        perm = rng.permutation(t)
        tr = perm[: t // 2]
        te = perm[t // 2 :]
        train_mean = np.mean(ret_matrix[tr, :], axis=0)
        best = int(np.argmax(train_mean))
        test_rank = int(np.argsort(np.argsort(np.mean(ret_matrix[te, :], axis=0)))[best]) + 1
        rel = test_rank / max(1, m)
        probs.append(rel < 0.5)
    return float(np.mean(np.asarray(probs, dtype=float)))

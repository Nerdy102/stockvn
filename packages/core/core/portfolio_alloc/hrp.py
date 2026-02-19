from __future__ import annotations

import numpy as np

from .covariance import sample_cov


def _ivp(cov: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(cov), 1e-12, None)
    inv = 1.0 / diag
    return inv / np.sum(inv)


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def _single_link_distance(a: list[int], b: list[int], dist: np.ndarray) -> float:
    return float(min(dist[i, j] for i in a for j in b))


def _agglomerative_single_link_order(dist: np.ndarray) -> list[int]:
    clusters: list[list[int]] = [[i] for i in range(dist.shape[0])]
    while len(clusters) > 1:
        best_i, best_j, best_d = 0, 1, float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _single_link_distance(clusters[i], clusters[j], dist)
                if d < best_d - 1e-15 or (abs(d - best_d) <= 1e-15 and (clusters[i][0], clusters[j][0]) < (clusters[best_i][0], clusters[best_j][0])):
                    best_i, best_j, best_d = i, j, d
        left = clusters[best_i]
        right = clusters[best_j]
        merged = left + right
        clusters = [c for k, c in enumerate(clusters) if k not in {best_i, best_j}] + [merged]
        clusters.sort(key=lambda c: c[0])
    return clusters[0]


def _leaf_order(corr: np.ndarray) -> list[int]:
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method="single")
        return [int(x) for x in leaves_list(z)]
    except Exception:
        return _agglomerative_single_link_order(dist)


def hrp_weights(returns_matrix) -> np.ndarray:
    x = np.asarray(returns_matrix, dtype=float)
    n = x.shape[1]
    if n == 1:
        return np.asarray([1.0])

    cov = sample_cov(x)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    order = _leaf_order(corr)

    w = {i: 1.0 for i in order}
    queue: list[list[int]] = [order]
    while queue:
        cluster = queue.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        var_l = _cluster_var(cov, left)
        var_r = _cluster_var(cov, right)
        alpha = 1.0 - var_l / max(var_l + var_r, 1e-12)
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1.0 - alpha)
        queue.append(left)
        queue.append(right)

    out = np.asarray([w[i] for i in range(n)], dtype=float)
    out = np.clip(out, 0.0, None)
    return out / np.sum(out)

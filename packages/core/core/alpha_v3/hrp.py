from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def _cluster_variance(cov: np.ndarray, cluster_indices: list[int]) -> float:
    """Cluster variance using inverse-variance portfolio inside the cluster."""
    sub_cov = cov[np.ix_(cluster_indices, cluster_indices)]
    diag = np.diag(sub_cov).astype(float)
    diag = np.where(diag <= 1e-12, 1e-12, diag)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def compute_hrp_weights(returns: np.ndarray) -> np.ndarray:
    """Compute long-only HRP weights from returns matrix (T, N).

    Steps:
    1) Ledoit-Wolf covariance estimate
    2) Correlation + distance = sqrt(0.5 * (1-corr))
    3) Single-linkage hierarchical clustering
    4) Quasi-diagonalization from dendrogram leaf order
    5) Recursive bisection cluster allocation
    """
    rets = np.asarray(returns, dtype=float)
    if rets.ndim != 2:
        raise ValueError("returns must have shape (T, N)")
    if rets.shape[1] == 0:
        raise ValueError("returns must contain at least one asset")

    # Keep estimator stable for occasional NaN/inf in upstream data.
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    cov = LedoitWolf().fit(rets).covariance_
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    corr = np.clip(cov / denom, -1.0, 1.0)

    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)

    if rets.shape[1] == 1:
        return np.array([1.0], dtype=float)

    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = list(leaves_list(link).astype(int))

    weights_ord = np.ones(len(order), dtype=float)
    clusters: list[list[int]] = [order]

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            var_total = var_left + var_right
            if var_total <= 1e-12:
                alloc_left = 0.5
            else:
                alloc_left = 1.0 - (var_left / var_total)
            alloc_left = float(np.clip(alloc_left, 0.0, 1.0))
            alloc_right = 1.0 - alloc_left

            left_mask = np.isin(order, left)
            right_mask = np.isin(order, right)
            weights_ord[left_mask] *= alloc_left
            weights_ord[right_mask] *= alloc_right

            next_clusters.extend([left, right])
        clusters = next_clusters

    weights_ord = np.maximum(weights_ord, 0.0)
    total = float(weights_ord.sum())
    if total <= 0:
        weights_ord = np.ones_like(weights_ord) / len(weights_ord)
    else:
        weights_ord /= total

    out = np.zeros(len(order), dtype=float)
    for pos, asset_idx in enumerate(order):
        out[asset_idx] = weights_ord[pos]
    out = np.maximum(out, 0.0)
    out /= out.sum()
    return out

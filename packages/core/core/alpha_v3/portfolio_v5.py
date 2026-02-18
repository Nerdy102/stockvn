from __future__ import annotations

from typing import Any

import numpy as np

from core.alpha_v3.hrp import compute_hrp_weights
from core.alpha_v3.portfolio import _normalize_positive, apply_cvar_overlay

PIPELINE_STEPS = ["hrp", "cvar_overlay", "strict_projection"]
MAX_SINGLE_CAP = 0.35
MAX_CLUSTER_CAP = 0.55
RISK_BUDGET_CAPS = {"low": 0.60, "mid": 0.35, "high": 0.20}
CASH_TARGET = 0.10


def _project_sum(x: np.ndarray, target_sum: float) -> np.ndarray:
    return np.asarray(x, dtype=float) + (float(target_sum) - float(np.sum(x))) / max(1, len(x))


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lower), upper)


def _project_group_cap(x: np.ndarray, idx: np.ndarray, cap: float) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    if len(idx) == 0:
        return out
    s = float(np.sum(out[idx]))
    if s <= float(cap):
        return out
    out[idx] *= float(cap) / max(s, 1e-12)
    return out


def strict_project_portfolio_v5(
    base_weights: np.ndarray,
    *,
    clusters: np.ndarray,
    risk_buckets: np.ndarray,
    target_sum: float,
    name_cap: float = MAX_SINGLE_CAP,
    cluster_cap: float = MAX_CLUSTER_CAP,
    risk_budget_caps: dict[str, float] | None = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    base = np.asarray(base_weights, dtype=float)
    if len(base) == 0:
        return np.array([], dtype=float)

    cluster_arr = np.asarray(clusters, dtype=object)
    risk_arr = np.asarray(risk_buckets, dtype=object)
    if len(cluster_arr) != len(base) or len(risk_arr) != len(base):
        raise ValueError("clusters/risk_buckets length must match base_weights")

    rb_caps = dict(RISK_BUDGET_CAPS)
    rb_caps.update(risk_budget_caps or {})

    lower = np.zeros_like(base)
    upper = np.full_like(base, float(name_cap), dtype=float)
    feasible_sum = min(float(target_sum), float(np.sum(upper)))

    x = _normalize_positive(base, target_sum=feasible_sum)
    x = _project_box(x, lower, upper)

    cluster_keys = sorted({str(v) for v in cluster_arr})
    risk_keys = sorted({str(v) for v in risk_arr})
    cluster_indices = [np.where(cluster_arr.astype(str) == k)[0] for k in cluster_keys]
    risk_constraints = [(np.where(risk_arr.astype(str) == k)[0], float(rb_caps[k])) for k in risk_keys if k in rb_caps]

    n_constraints = 2 + len(cluster_indices) + len(risk_constraints)
    increments = [np.zeros_like(x) for _ in range(n_constraints)]

    for _ in range(max_iter):
        prev = x.copy()

        y = x + increments[0]
        x = _project_box(y, lower, upper)
        increments[0] = y - x

        y = x + increments[1]
        x = _project_sum(y, feasible_sum)
        increments[1] = y - x

        cur = 2
        for idx in cluster_indices:
            y = x + increments[cur]
            x = _project_group_cap(y, idx, cluster_cap)
            increments[cur] = y - x
            cur += 1

        for idx, cap in risk_constraints:
            y = x + increments[cur]
            x = _project_group_cap(y, idx, cap)
            increments[cur] = y - x
            cur += 1

        if np.linalg.norm(x - prev, ord=2) <= tol:
            break

    for _ in range(30):
        x = _project_box(x, lower, upper)
        for idx in cluster_indices:
            x = _project_group_cap(x, idx, cluster_cap)
        for idx, cap in risk_constraints:
            x = _project_group_cap(x, idx, cap)
        x = _project_sum(x, feasible_sum)

    return _project_box(x, lower, upper)


def _violations(
    w: np.ndarray,
    *,
    target_sum: float,
    clusters: np.ndarray,
    risk_buckets: np.ndarray,
    name_cap: float,
    cluster_cap: float,
    risk_budget_caps: dict[str, float],
) -> dict[str, float]:
    vec = np.asarray(w, dtype=float)
    c = np.asarray(clusters, dtype=object)
    r = np.asarray(risk_buckets, dtype=object)

    cluster_excess = 0.0
    for key in sorted({str(v) for v in c}):
        idx = np.where(c.astype(str) == key)[0]
        cluster_excess = max(cluster_excess, max(0.0, float(np.sum(vec[idx])) - float(cluster_cap)))

    risk_excess = 0.0
    for key, cap in sorted(risk_budget_caps.items()):
        idx = np.where(r.astype(str) == str(key))[0]
        if len(idx) == 0:
            continue
        risk_excess = max(risk_excess, max(0.0, float(np.sum(vec[idx])) - float(cap)))

    return {
        "nonneg": float(np.max(np.maximum(0.0, -vec))),
        "sum": float(abs(float(np.sum(vec)) - float(target_sum))),
        "name_cap": float(np.max(np.maximum(0.0, vec - float(name_cap)))),
        "cluster_cap": float(cluster_excess),
        "risk_budget_cap": float(risk_excess),
    }


def build_portfolio_v5(
    returns_252: np.ndarray,
    *,
    clusters: np.ndarray,
    risk_buckets: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    risky_target = 1.0 - CASH_TARGET

    w_hrp = compute_hrp_weights(returns_252)
    w_cvar, _cvar_info = apply_cvar_overlay(returns_252, w_hrp, alpha=0.05, lower_mult=0.5, upper_mult=1.5)
    w_cvar = _normalize_positive(np.maximum(w_cvar, 0.0), target_sum=risky_target)

    w_proj = strict_project_portfolio_v5(
        w_cvar,
        clusters=np.asarray(clusters, dtype=object),
        risk_buckets=np.asarray(risk_buckets, dtype=object),
        target_sum=risky_target,
        name_cap=MAX_SINGLE_CAP,
        cluster_cap=MAX_CLUSTER_CAP,
        risk_budget_caps=RISK_BUDGET_CAPS,
    )

    cash = max(0.0, 1.0 - float(np.sum(w_proj)))

    pre = _violations(
        w_cvar,
        target_sum=risky_target,
        clusters=np.asarray(clusters, dtype=object),
        risk_buckets=np.asarray(risk_buckets, dtype=object),
        name_cap=MAX_SINGLE_CAP,
        cluster_cap=MAX_CLUSTER_CAP,
        risk_budget_caps=RISK_BUDGET_CAPS,
    )
    post = _violations(
        w_proj,
        target_sum=risky_target,
        clusters=np.asarray(clusters, dtype=object),
        risk_buckets=np.asarray(risk_buckets, dtype=object),
        name_cap=MAX_SINGLE_CAP,
        cluster_cap=MAX_CLUSTER_CAP,
        risk_budget_caps=RISK_BUDGET_CAPS,
    )

    report = {
        "schema_version": "portfolio_build_v5",
        "pipeline_steps": list(PIPELINE_STEPS),
        "caps": {"name_cap": float(MAX_SINGLE_CAP), "cluster_cap": float(MAX_CLUSTER_CAP)},
        "risk_budgets": {k: float(v) for k, v in sorted(RISK_BUDGET_CAPS.items())},
        "targets": {"risky_sum": float(risky_target), "cash_target": float(CASH_TARGET)},
        "violations_pre": pre,
        "violations_post": post,
        "active_constraints": ["nonneg", "sum_risky", "name_cap", "cluster_cap", "risk_budget_cap"],
        "feasible": bool(max(post.values()) <= 1e-8),
    }
    return w_proj, cash, report

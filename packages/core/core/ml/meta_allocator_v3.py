from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UTILITY_CVAR_MULT = 3.0
UTILITY_COST_MULT = 0.5
EG_ETA = 8.0
WEIGHT_CAP_MIN = 0.05
WEIGHT_CAP_MAX = 0.80


@dataclass(frozen=True)
class MetaAllocatorV3Config:
    eta: float = EG_ETA
    min_weight: float = WEIGHT_CAP_MIN
    max_weight: float = WEIGHT_CAP_MAX


def compute_expert_utility_v3(experts: pd.DataFrame) -> pd.Series:
    r21 = pd.to_numeric(experts["r21"], errors="coerce").fillna(0.0)
    cvar = pd.to_numeric(experts["cvar"], errors="coerce").fillna(0.0)
    cost_bps = pd.to_numeric(experts["cost_bps"], errors="coerce").fillna(0.0)
    return r21 - (UTILITY_CVAR_MULT * cvar) - (UTILITY_COST_MULT * (cost_bps / 10000.0))


def _project_with_caps(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    w = np.array(weights, dtype=float)
    n = len(w)
    if n == 0:
        return w

    lower = np.full(n, float(min_weight), dtype=float)
    upper = np.full(n, float(max_weight), dtype=float)
    if lower.sum() - 1.0 > 1e-12 or upper.sum() + 1e-12 < 1.0:
        raise ValueError("infeasible caps for simplex projection")

    w = np.clip(w, lower, upper)
    for _ in range(50):
        delta = 1.0 - float(w.sum())
        if abs(delta) <= 1e-12:
            break
        if delta > 0:
            room = upper - w
        else:
            room = w - lower
        active = room > 1e-12
        if not np.any(active):
            break
        step = delta / float(room[active].size)
        w[active] = w[active] + np.sign(delta) * np.minimum(abs(step), room[active])
    return w / max(w.sum(), 1e-12)


def meta_allocate_v3(
    experts: pd.DataFrame,
    prev_weights: pd.Series | None = None,
    *,
    config: MetaAllocatorV3Config | None = None,
) -> tuple[pd.Series, dict[str, object]]:
    cfg = config or MetaAllocatorV3Config()
    cols = ["expert", "r21", "cvar", "cost_bps"]
    missing = [c for c in cols if c not in experts.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    work = experts[cols].copy()
    work["expert"] = work["expert"].astype(str)
    utility = compute_expert_utility_v3(work)

    n = len(work)
    if n == 0:
        return pd.Series(dtype=float), {"experts": [], "components": []}

    if prev_weights is None:
        base = np.full(n, 1.0 / n, dtype=float)
    else:
        aligned = prev_weights.reindex(work["expert"]).fillna(0.0).to_numpy(dtype=float)
        base = aligned / max(float(aligned.sum()), 1e-12)

    raw = base * np.exp(cfg.eta * utility.to_numpy(dtype=float))
    raw = raw / max(float(raw.sum()), 1e-12)
    bounded = _project_with_caps(raw, min_weight=cfg.min_weight, max_weight=cfg.max_weight)

    weights = pd.Series(bounded, index=work["expert"], dtype=float)
    audit_rows: list[dict[str, float | str]] = []
    for idx, row in work.reset_index(drop=True).iterrows():
        cost_frac = float(row["cost_bps"]) / 10000.0
        audit_rows.append(
            {
                "expert": str(row["expert"]),
                "r21": float(row["r21"]),
                "cvar": float(row["cvar"]),
                "cost_bps": float(row["cost_bps"]),
                "cost_frac": cost_frac,
                "utility": float(utility.iloc[idx]),
                "base_weight": float(base[idx]),
                "eg_weight_raw": float(raw[idx]),
                "weight": float(bounded[idx]),
            }
        )

    audit = {
        "formula": "u = r21 - 3*cvar - 0.5*(cost_bps/10000)",
        "eta": float(cfg.eta),
        "caps": {"min_weight": float(cfg.min_weight), "max_weight": float(cfg.max_weight)},
        "components": audit_rows,
    }
    return weights, audit

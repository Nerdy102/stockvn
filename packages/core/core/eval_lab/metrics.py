from __future__ import annotations

import numpy as np


def compute_tail_metrics(returns: list[float]) -> dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return {"var95": 0.0, "es95": 0.0}
    var95 = float(np.quantile(arr, 0.05))
    es95 = float(np.mean(arr[arr <= var95])) if np.any(arr <= var95) else var95
    return {"var95": var95, "es95": es95}


def compute_performance_metrics(
    returns: list[float], equity: list[float], turnover: list[float]
) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    e = np.asarray(equity, dtype=float)
    t = np.asarray(turnover, dtype=float)
    if r.size == 0:
        return {
            k: 0.0
            for k in [
                "total_return",
                "cagr",
                "annual_vol",
                "sharpe",
                "sortino",
                "calmar",
                "mdd",
                "turnover_l1",
            ]
        }
    total = float(e[-1] - 1.0)
    cagr = float(e[-1] ** (252.0 / max(1, r.size)) - 1.0)
    vol = float(np.std(r) * np.sqrt(252.0))
    sharpe = float(np.mean(r) / max(1e-8, np.std(r)) * np.sqrt(252.0))
    dn = r[r < 0]
    sortino = float(np.mean(r) / max(1e-8, np.std(dn) if dn.size else 1e-8) * np.sqrt(252.0))
    mdd = float(np.min(e / np.maximum.accumulate(e) - 1.0))
    calmar = float(cagr / max(1e-8, abs(mdd)))
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "turnover_l1": float(np.sum(np.abs(t))),
    }


def compute_cost_attribution(gross_end: float, net_end: float) -> dict[str, float]:
    drag = (gross_end - net_end) / max(1e-8, gross_end)
    return {"cost_drag_pct": float(drag)}

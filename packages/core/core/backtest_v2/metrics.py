from __future__ import annotations

import math


def compute_metrics(nav: list[float]) -> dict[str, float]:
    if not nav:
        return {
            "net_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "turnover": 0.0,
        }
    rets = []
    for i in range(1, len(nav)):
        rets.append(nav[i] / max(nav[i - 1], 1e-9) - 1.0)
    net_return = nav[-1] / max(nav[0], 1e-9) - 1.0
    peak = nav[0]
    mdd = 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / max(peak, 1e-9) - 1.0)
    mu = sum(rets) / max(len(rets), 1)
    var = sum((r - mu) ** 2 for r in rets) / max(len(rets), 1)
    vol = math.sqrt(var)
    sharpe = (mu / vol * math.sqrt(252)) if vol > 0 else 0.0
    downside = [min(r, 0.0) for r in rets]
    dvar = sum((r - 0.0) ** 2 for r in downside) / max(len(downside), 1)
    dvol = math.sqrt(dvar)
    sortino = (mu / dvol * math.sqrt(252)) if dvol > 0 else 0.0
    cagr = (nav[-1] / max(nav[0], 1e-9)) ** (252 / max(len(nav), 1)) - 1.0
    return {
        "net_return": float(net_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "turnover": 0.0,
    }

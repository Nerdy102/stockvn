from __future__ import annotations

import numpy as np


STRICT_TOL = 1e-6


def check_equity_identity(
    equity_gross: list[float],
    equity_net: list[float],
    cum_cost: list[float],
    tol: float = STRICT_TOL,
) -> bool:
    if not equity_gross:
        return True
    g = np.asarray(equity_gross, dtype=float)
    n = np.asarray(equity_net, dtype=float)
    c = np.asarray(cum_cost, dtype=float)
    err = np.max(np.abs((g - n) - c))
    return bool(err < tol)


def check_end_identity(
    gross_end: float, net_end: float, total_cost_total: float, tol: float = STRICT_TOL
) -> bool:
    return abs((float(gross_end) - float(net_end)) - float(total_cost_total)) < tol


def check_return_identity(equity_net: list[float], total_return_reported: float) -> bool:
    if not equity_net:
        return abs(total_return_reported) < 1e-12
    implied = float(equity_net[-1] - 1.0)
    return abs(implied - float(total_return_reported)) < 1e-9


def check_cost_nonnegativity(cost_components: dict[str, float]) -> bool:
    return all(float(v) >= -1e-12 for v in cost_components.values())

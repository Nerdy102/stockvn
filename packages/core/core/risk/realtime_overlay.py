from __future__ import annotations

from typing import Any


def evaluate_intraday_killswitch(
    *,
    nav: float,
    intraday_drawdown: float,
    expected_daily_vol_proxy: float,
) -> dict[str, Any]:
    nav_v = max(float(nav), 0.0)
    dd_v = max(float(intraday_drawdown), 0.0)
    vol_v = max(float(expected_daily_vol_proxy), 0.0)
    threshold = 1.5 * vol_v * nav_v
    paused = dd_v > threshold if nav_v > 0 else False
    return {
        "paused": paused,
        "threshold": threshold,
        "intraday_drawdown": dd_v,
        "expected_daily_vol_proxy": vol_v,
        "reason": "intraday_drawdown_killswitch" if paused else "",
    }

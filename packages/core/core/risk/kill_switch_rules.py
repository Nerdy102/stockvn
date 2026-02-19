from __future__ import annotations

import datetime as dt


def evaluate_kill_switch(
    *,
    daily_loss_pct: float,
    drawdown_pct: float,
    as_of: dt.datetime,
    market: str,
    now: dt.datetime | None = None,
    max_daily_loss_pct: float = 0.02,
    max_drawdown_pct: float = 0.1,
) -> tuple[str, str | None]:
    ref = now or dt.datetime.utcnow()
    if daily_loss_pct >= max_daily_loss_pct:
        return "PAUSED", "RISK_DAILY_LOSS_BREACH"
    if drawdown_pct >= max_drawdown_pct:
        return "PAUSED", "RISK_DRAWDOWN_BREACH"
    if market == "vn" and (ref.date() - as_of.date()).days > 2:
        return "PAUSED", "DATA_STALE"
    if market == "crypto" and (ref - as_of).total_seconds() > 6 * 3600:
        return "PAUSED", "DATA_STALE"
    return "RUNNING", None

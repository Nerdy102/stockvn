from __future__ import annotations

import datetime as dt

from core.risk.kill_switch_rules import evaluate_kill_switch


def test_kill_switch_daily_loss() -> None:
    state, reason = evaluate_kill_switch(
        daily_loss_pct=0.03,
        drawdown_pct=0.01,
        as_of=dt.datetime.utcnow(),
        market="vn",
    )
    assert state == "PAUSED"
    assert reason == "RISK_DAILY_LOSS_BREACH"

from __future__ import annotations

from core.risk_overlay import evaluate_intraday_killswitch


def test_killswitch_pause_when_drawdown_exceeds_threshold() -> None:
    out = evaluate_intraday_killswitch(
        nav=1_000_000_000,
        intraday_drawdown=25_000_000,
        expected_daily_vol_proxy=0.01,
    )
    assert out["paused"] is True
    assert out["reason"] == "intraday_drawdown_killswitch"

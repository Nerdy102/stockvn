from __future__ import annotations

from core.monitoring.drift_monitor import evaluate_drift_pause


def test_drift_pause_conditions() -> None:
    trades = []
    for _ in range(30):
        trades.append({"realized_slippage": 30, "est_slippage": 10, "pnl": -1, "mdd": 0.01})
    alerts = evaluate_drift_pause(model_id="model_1", market="vn", recent_trades=trades)
    codes = {a.code for a in alerts}
    assert "DRIFT_SLIPPAGE_RATIO_HIGH" in codes

from __future__ import annotations

from core.monitoring.performance_drift import detect_slippage_anomaly


def test_drift_slippage_anomaly() -> None:
    state, reason = detect_slippage_anomaly(
        slippage_est=[1.0] * 15,
        slippage_real=[3.0] * 15,
    )
    assert state == "PAUSED"
    assert reason == "SLIPPAGE_ANOMALY"

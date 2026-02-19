from __future__ import annotations


def detect_slippage_anomaly(
    slippage_est: list[float], slippage_real: list[float]
) -> tuple[str, str | None]:
    if len(slippage_est) < 10 or len(slippage_real) < 10:
        return "RUNNING", None
    est = sum(slippage_est[-10:]) / 10.0
    real = sum(slippage_real[-10:]) / 10.0
    if est > 0 and real > 2.0 * est:
        return "PAUSED", "SLIPPAGE_ANOMALY"
    return "RUNNING", None

from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def test_gates_fail_returns_neutral() -> None:
    df = pd.DataFrame([
        {"date": "2025-01-01", "open": 10, "high": 9, "low": 11, "close": 0, "volume": -1},
    ])
    sig = run_signal("model_1", "FPT", "1D", df)
    assert sig.signal == "TRUNG_TINH"
    assert sig.confidence_bucket == "Thấp"
    assert "mã lỗi" in sig.reason_short
    assert sig.debug_fields.get("degraded_ok") is True

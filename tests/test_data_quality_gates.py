from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def _bad_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2025-01-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000},
            {"date": "2025-01-01", "open": 10, "high": 9, "low": 8, "close": 10, "volume": -1},
            {"date": "2025-01-03", "open": 0, "high": 10, "low": 9, "close": 10, "volume": 1000},
        ]
    )


def test_data_quality_gate_fail_neutral_signal() -> None:
    sig = run_signal("model_1", "FPT", "1D", _bad_df())
    assert sig.signal == "TRUNG_TINH"
    assert "Thiếu/không hợp lệ dữ liệu để phân tích" in sig.reason_short

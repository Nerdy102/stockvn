from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def test_risk_tags_max_2() -> None:
    rows = []
    close = 100.0
    for i in range(220):
        if i == 219:
            close = close * 1.5
        else:
            close += 0.1
        rows.append({
            "date": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 0 if i == 219 else 1_000_000,
        })
    sig = run_signal("model_3", "BTCUSDT", "1D", pd.DataFrame(rows), market="crypto")
    assert len(sig.risk_tags) <= 2

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "realtime_signal_engine"))

from realtime_signal_engine.evaluator import evaluate_alert_dsl_on_bar_close


def test_alert_dsl_eval_on_bar_close() -> None:
    df = pd.DataFrame(
        [
            {"close": 10.0, "EMA20": 10.2},
            {"close": 11.0, "EMA20": 10.2},
        ]
    )
    assert evaluate_alert_dsl_on_bar_close(df, "close > EMA20") is True

from __future__ import annotations

import pandas as pd

from core.alpha_v3.calibration import summarize_reset_events


def test_reset_events_sorted() -> None:
    events = pd.DataFrame(
        [
            {"date": "2025-01-03", "event_type": "reset", "before_coverage": 0.7, "after_coverage": 0.82},
            {"date": "2025-01-01", "event_type": "reset", "before_coverage": 0.75, "after_coverage": 0.8},
        ]
    )
    out = summarize_reset_events(events)
    assert [r["date"] for r in out] == ["2025-01-01", "2025-01-03"]

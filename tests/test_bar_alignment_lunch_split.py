from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))

from bar_builder.bar_builder import build_bar_windows


def test_bar_windows_do_not_span_lunch_break() -> None:
    d = dt.date(2025, 1, 2)
    bars15 = build_bar_windows(d, "HOSE", "15m")
    bars60 = build_bar_windows(d, "HOSE", "60m")

    # 11:30-13:00 local = 04:30-06:00 UTC should not be crossed by any bar.
    lunch_start = dt.datetime(2025, 1, 2, 4, 30, tzinfo=dt.timezone.utc)
    lunch_end = dt.datetime(2025, 1, 2, 6, 0, tzinfo=dt.timezone.utc)

    for b in bars15 + bars60:
        assert not (b.start_ts < lunch_start and b.end_ts > lunch_start)
        assert not (b.start_ts < lunch_end and b.end_ts > lunch_end)

    assert any((b.end_ts - b.start_ts).total_seconds() == 15 * 60 for b in bars15)
    assert any((b.end_ts - b.start_ts).total_seconds() < 60 * 60 for b in bars60)

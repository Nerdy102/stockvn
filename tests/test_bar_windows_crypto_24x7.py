from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))

from bar_builder.bar_builder import build_bar_windows


def test_crypto_bar_windows_cover_full_day_utc() -> None:
    d = dt.date(2025, 1, 2)
    bars15 = build_bar_windows(d, "CRYPTO", "15m")
    bars60 = build_bar_windows(d, "CRYPTO", "60m")

    assert len(bars15) == 96
    assert len(bars60) == 24

    assert bars15[0].start_ts == dt.datetime(2025, 1, 2, 0, 0, tzinfo=dt.timezone.utc)
    assert bars15[-1].end_ts == dt.datetime(2025, 1, 3, 0, 0, tzinfo=dt.timezone.utc)

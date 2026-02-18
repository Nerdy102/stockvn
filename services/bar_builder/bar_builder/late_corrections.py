from __future__ import annotations

from typing import Any


def build_correction_event(
    *,
    symbol: str,
    timeframe: str,
    bar_start_ts: str,
    bar_end_ts: str,
    reason: str,
    original_event_id: str,
) -> dict[str, Any]:
    return {
        "event_type": "CORRECTION",
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_start_ts": bar_start_ts,
        "bar_end_ts": bar_end_ts,
        "reason": reason,
        "original_event_id": original_event_id,
    }

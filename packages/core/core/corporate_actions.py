from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

log = logging.getLogger(__name__)


def adjust_prices(
    symbol: str, bars: pd.DataFrame, start: dt.date, end: dt.date, method: str = "none"
) -> pd.DataFrame:
    """Hook for corporate action adjustments (MVP warning-only)."""
    out = bars.copy()
    if method == "none":
        return out
    log.warning(
        "Corporate action adjustment hook called but method is not fully implemented.",
        extra={"symbol": symbol, "method": method, "start": str(start), "end": str(end)},
    )
    out["is_adjusted"] = False
    return out

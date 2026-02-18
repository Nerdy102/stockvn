from __future__ import annotations

import datetime as dt
import logging
import math

import pandas as pd
from core.calendar_vn import get_trading_calendar_vn

DAILY_MAX_DAYS_DEFAULT = 365
INTRADAY_MAX_TRADING_DAYS_DEFAULT = 30
MAX_POINTS_PER_CHART = 10_000


def _trading_days_count(start: dt.date, end: dt.date) -> int:
    cal = get_trading_calendar_vn()
    return len(cal.trading_days_between(start, end, inclusive="both"))


def enforce_bounded_range(start: dt.date, end: dt.date, max_days: int, page_id: str = "") -> None:
    days = _trading_days_count(start, end)
    if days > max_days:
        raise ValueError(
            f"Khoảng thời gian đang chọn là {days} ngày giao dịch, vượt ngân sách {max_days}. "
            "Vui lòng thu hẹp khoảng ngày để giao diện phản hồi ổn định."
        )


def enforce_intraday_default_window(
    timeframe: str,
    start: dt.date,
    end: dt.date,
    page_id: str = "",
) -> tuple[dt.date, dt.date]:
    if timeframe not in {"15m", "60m"}:
        return start, end
    days = _trading_days_count(start, end)
    if days <= INTRADAY_MAX_TRADING_DAYS_DEFAULT:
        return start, end
    cal = get_trading_calendar_vn()
    clamped_start = cal.shift_trading_days(end, -(INTRADAY_MAX_TRADING_DAYS_DEFAULT - 1))
    logging.info(
        "intraday range clamped",
        extra={"page_id": page_id, "timeframe": timeframe, "start": str(start), "end": str(end)},
    )
    return clamped_start, end


def enforce_point_budget(n_points: int) -> None:
    if n_points > MAX_POINTS_PER_CHART:
        raise ValueError(
            f"Số điểm dữ liệu {n_points:,} vượt ngân sách {MAX_POINTS_PER_CHART:,}. "
            "Vui lòng downsample trước khi vẽ biểu đồ."
        )


def downsample_df(df: pd.DataFrame, max_points: int, page_id: str = "") -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    stride = math.ceil(len(df) / max_points)
    logging.info(
        "downsampling applied",
        extra={
            "page_id": page_id,
            "n_before": len(df),
            "n_after": len(df.iloc[::stride]),
            "stride": stride,
        },
    )
    return df.iloc[::stride].copy()

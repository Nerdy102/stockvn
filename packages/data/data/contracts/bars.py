from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC_TZ = ZoneInfo("UTC")


class BarRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    market: Literal["vn", "crypto"]
    timeframe: Literal["1D", "60m"]
    ts: dt.datetime
    open: Decimal | float
    high: Decimal | float
    low: Decimal | float
    close: Decimal | float
    volume: Decimal | float
    vwap: Decimal | float | None = None
    quote_volume: Decimal | float | None = None
    trades_count: int | None = Field(default=None, ge=0)
    provider: str | None = None
    provider_event_id: str | None = None

    @field_validator("ts")
    @classmethod
    def _ts_must_be_aware_utc(cls, v: dt.datetime) -> dt.datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC_TZ)
        return v.astimezone(UTC_TZ)


def normalize_bar_timestamp(ts: object) -> dt.datetime:
    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError("timestamp không hợp lệ")
    return parsed.to_pydatetime().astimezone(UTC_TZ)


def display_timestamp_for_market(ts_utc: dt.datetime, market: str) -> dt.datetime:
    aware_utc = ts_utc if ts_utc.tzinfo is not None else ts_utc.replace(tzinfo=UTC_TZ)
    aware_utc = aware_utc.astimezone(UTC_TZ)
    if market == "vn":
        return aware_utc.astimezone(VN_TZ)
    return aware_utc

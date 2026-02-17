from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CanonicalBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Ticker(CanonicalBaseModel):
    symbol: str
    exchange: str
    instrument_type: str = "stock"
    sector: str | None = None
    industry: str | None = None
    tags: list[str] = Field(default_factory=list)
    first_trading_date: date | None = None
    last_trading_date: date | None = None
    status: str = "active"


class Bar(CanonicalBaseModel):
    symbol: str
    timeframe: str
    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    value: float | None = None
    is_adjusted: bool = False
    data_source: str = "unknown"
    ingest_ts: datetime | None = None


class QuoteLevel(CanonicalBaseModel):
    symbol: str
    ts_utc: datetime
    bid_prices: list[float] = Field(default_factory=list)
    bid_volumes: list[float] = Field(default_factory=list)
    ask_prices: list[float] = Field(default_factory=list)
    ask_volumes: list[float] = Field(default_factory=list)


class TradePrint(CanonicalBaseModel):
    symbol: str
    ts_utc: datetime
    last_price: float
    last_vol: float
    total_val: float | None = None
    total_vol: float | None = None
    side: str | None = None


class ForeignRoomSnapshot(CanonicalBaseModel):
    symbol: str
    ts_utc: datetime
    total_room: float | None = None
    current_room: float | None = None
    buy_vol: float | None = None
    sell_vol: float | None = None
    buy_val: float | None = None
    sell_val: float | None = None


class IndexSnapshot(CanonicalBaseModel):
    index_id: str
    ts_utc: datetime
    index_value: float
    change: float | None = None
    ratio_change: float | None = None
    advances: int | None = None
    declines: int | None = None
    nochange: int | None = None
    ceilings: int | None = None
    floors: int | None = None
    total_value: float | None = None
    all_value: float | None = None
    session: str | None = None


class FundamentalsPointInTime(CanonicalBaseModel):
    symbol: str
    period_end: date
    public_date: date
    as_of_date: date
    statement_type: Literal["quarterly", "yearly", "other"]
    public_date_is_assumed: bool = False


class CorporateAction(CanonicalBaseModel):
    symbol: str
    action_type: str
    ex_date: date | None = None
    record_date: date | None = None
    payment_date: date | None = None
    ratio: float | None = None
    cash_amount: float | None = None
    adj_factor: float | None = None
    note: str | None = None
    source: str | None = None


class OddLotSnapshot(CanonicalBaseModel):
    symbol: str
    ts_utc: datetime
    last_price: float | None = None
    last_vol: float | None = None
    total_val: float | None = None
    total_vol: float | None = None
    bid_prices: list[float] = Field(default_factory=list)
    bid_volumes: list[float] = Field(default_factory=list)
    ask_prices: list[float] = Field(default_factory=list)
    ask_volumes: list[float] = Field(default_factory=list)
    trading_status: str | None = None
    trading_session: str | None = None

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from sqlalchemy import BigInteger, Column, Index, JSON, String
from sqlmodel import Field, SQLModel


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


JsonDict = Dict[str, Any]


class Ticker(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    name: str
    exchange: str = Field(index=True)
    sector: str = Field(index=True)
    industry: str

    shares_outstanding: int = Field(default=0, sa_column=Column(BigInteger))
    market_cap: float = 0.0
    is_bank: bool = False
    is_broker: bool = False
    tags: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)
    updated_at: dt.datetime = Field(default_factory=utcnow)


class PriceOHLCV(SQLModel, table=True):
    __table_args__ = (
        Index("ix_prices_timeframe_timestamp", "timeframe", "timestamp"),
        Index("ix_prices_timestamp", "timestamp"),
    )

    symbol: str = Field(primary_key=True, index=True)
    timeframe: str = Field(primary_key=True, index=True)
    timestamp: dt.datetime = Field(primary_key=True)

    open: float
    high: float
    low: float
    close: float
    volume: float
    value_vnd: float = 0.0
    source: str = Field(default="legacy", index=True)
    quality_flags: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class BronzeRaw(SQLModel, table=True):
    __table_args__ = (
        Index("ix_bronze_received_at", "received_at"),
        Index("ix_bronze_symbol", "symbol"),
        Index("ix_bronze_provider_endpoint_hash", "provider_name", "endpoint_or_channel", "payload_hash", unique=True),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    provider_name: str = Field(index=True)
    endpoint_or_channel: str = Field(index=True)
    received_at: dt.datetime = Field(default_factory=utcnow)
    trading_date: Optional[dt.date] = None
    symbol: Optional[str] = Field(default=None, index=True)
    index_id: Optional[str] = Field(default=None, index=True)
    payload_hash: str = Field(index=True)
    raw_payload: str = Field(sa_column=Column(String))
    schema_version: str = Field(default="v1")


class IngestState(SQLModel, table=True):
    provider: str = Field(primary_key=True)
    channel: str = Field(primary_key=True)
    symbol: str = Field(primary_key=True)
    last_ts: Optional[dt.datetime] = None
    last_cursor: Optional[str] = None
    updated_at: dt.datetime = Field(default_factory=utcnow)


class QuoteL2(SQLModel, table=True):
    __table_args__ = (Index("ix_quotes_symbol_ts", "symbol", "timestamp"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    bids: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    asks: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = Field(default="ssi_fcdata")


class TradeTape(SQLModel, table=True):
    __table_args__ = (Index("ix_trades_symbol_ts", "symbol", "timestamp"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    last_price: float
    last_vol: float
    side: Optional[str] = None
    source: str = Field(default="ssi_fcdata")


class ForeignRoom(SQLModel, table=True):
    __table_args__ = (Index("ix_foreign_symbol_ts", "symbol", "timestamp"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    total_room: Optional[float] = None
    current_room: Optional[float] = None
    fbuy_vol: Optional[float] = None
    fsell_vol: Optional[float] = None
    fbuy_val: Optional[float] = None
    fsell_val: Optional[float] = None
    source: str = Field(default="ssi_fcdata")


class IndexOHLCV(SQLModel, table=True):
    __table_args__ = (Index("ix_index_ohlcv_id_tf_ts", "index_id", "timeframe", "timestamp", unique=True),)

    id: Optional[int] = Field(default=None, primary_key=True)
    index_id: str = Field(index=True)
    timeframe: str = Field(default="1D", index=True)
    timestamp: dt.datetime = Field(index=True)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    value: Optional[float] = None
    volume: Optional[float] = None
    source: str = Field(default="ssi_fcdata")


class IndicatorState(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    timeframe: str = Field(primary_key=True)
    indicator_name: str = Field(primary_key=True)
    state_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: dt.datetime = Field(default_factory=utcnow)


class Fundamental(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    as_of_date: dt.date = Field(primary_key=True)
    public_date: Optional[dt.date] = None
    sector: str
    is_bank: bool = False
    is_broker: bool = False

    revenue_ttm_vnd: float = 0.0
    net_income_ttm_vnd: float = 0.0
    gross_profit_ttm_vnd: float = 0.0
    ebitda_ttm_vnd: float = 0.0
    cfo_ttm_vnd: float = 0.0
    dividends_ttm_vnd: float = 0.0

    total_assets_vnd: float = 0.0
    total_liabilities_vnd: float = 0.0
    equity_vnd: float = 0.0
    net_debt_vnd: float = 0.0

    nim: Optional[float] = None
    casa: Optional[float] = None
    cir: Optional[float] = None
    npl_ratio: Optional[float] = None
    llr_coverage: Optional[float] = None
    credit_growth: Optional[float] = None
    car: Optional[float] = None
    margin_lending_vnd: Optional[float] = None
    adtv_sensitivity: Optional[float] = None
    proprietary_gains_ratio: Optional[float] = None


class CorporateAction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    action_type: str
    ex_date: Optional[dt.date] = None
    record_date: Optional[dt.date] = None
    pay_date: Optional[dt.date] = None
    amount: Optional[float] = None
    adjust_method: str = "none"
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class ScreenResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    screen_name: str
    run_at: dt.datetime = Field(default_factory=utcnow)
    params: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    results: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class IndicatorValue(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    timeframe: str = Field(primary_key=True)
    timestamp: dt.datetime = Field(primary_key=True)
    name: str = Field(primary_key=True)
    value: float


class FactorScore(SQLModel, table=True):
    __table_args__ = (
        Index("ix_factor_asof", "as_of_date"),
        Index("ix_factor_symbol_asof", "symbol", "as_of_date"),
    )

    symbol: str = Field(primary_key=True)
    as_of_date: dt.date = Field(primary_key=True)
    factor: str = Field(primary_key=True)
    score: float
    raw: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class Signal(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    timeframe: str = Field(primary_key=True)
    timestamp: dt.datetime = Field(primary_key=True)
    signal_type: str = Field(primary_key=True)
    strength: float = 1.0
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class AlertRule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    timeframe: str = "1D"
    expression: str
    symbols_csv: str = ""
    is_active: bool = True
    created_at: dt.datetime = Field(default_factory=utcnow)


class AlertEvent(SQLModel, table=True):
    rule_id: int = Field(primary_key=True)
    symbol: str = Field(primary_key=True)
    triggered_at: dt.datetime = Field(primary_key=True)
    message: str
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class Portfolio(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: dt.datetime = Field(default_factory=utcnow)


class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int
    trade_date: dt.date
    symbol: str
    side: str
    quantity: float
    price: float
    strategy_tag: str = ""
    notes: str = ""
    commission: float = 0.0
    taxes: float = 0.0
    external_id: str = Field(index=True)

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from sqlalchemy import Column, JSON, BigInteger
from sqlmodel import Field, SQLModel


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


JsonDict = Dict[str, Any]


class Ticker(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    name: str
    exchange: str
    sector: str
    industry: str

    # FIX: dùng BIGINT để tránh lỗi integer out of range (vd 2,400,000,000)
    shares_outstanding: int = Field(default=0, sa_column=Column(BigInteger))

    market_cap: float = 0.0
    is_bank: bool = False
    is_broker: bool = False
    tags: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)
    updated_at: dt.datetime = Field(default_factory=utcnow)


class PriceOHLCV(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    timeframe: str = Field(primary_key=True)  # 1D, 60m, 15m
    timestamp: dt.datetime = Field(primary_key=True)

    open: float
    high: float
    low: float
    close: float
    volume: float
    value_vnd: float = 0.0


class Fundamental(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    as_of_date: dt.date = Field(primary_key=True)
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

    # Banks
    nim: Optional[float] = None
    casa: Optional[float] = None
    cir: Optional[float] = None
    npl_ratio: Optional[float] = None
    llr_coverage: Optional[float] = None
    credit_growth: Optional[float] = None
    car: Optional[float] = None

    # Brokers
    margin_lending_vnd: Optional[float] = None
    adtv_sensitivity: Optional[float] = None
    proprietary_gains_ratio: Optional[float] = None


class CorporateAction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    action_type: str  # dividend|split|rights|...
    ex_date: Optional[dt.date] = None
    record_date: Optional[dt.date] = None
    pay_date: Optional[dt.date] = None
    amount: Optional[float] = None
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
    symbol: str = Field(primary_key=True)
    as_of_date: dt.date = Field(primary_key=True)
    factor: str = Field(primary_key=True)  # value|quality|...|total
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
    side: str  # BUY/SELL
    quantity: float
    price: float
    strategy_tag: str = ""
    notes: str = ""
    commission: float = 0.0
    taxes: float = 0.0
    external_id: str = Field(index=True)

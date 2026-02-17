from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, BigInteger, Column, Index, String
from sqlmodel import Field, SQLModel


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


JsonDict = dict[str, Any]


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
        Index(
            "ix_bronze_provider_endpoint_hash",
            "provider_name",
            "endpoint_or_channel",
            "payload_hash",
            unique=True,
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider_name: str = Field(index=True)
    endpoint_or_channel: str = Field(index=True)
    received_at: dt.datetime = Field(default_factory=utcnow)
    trading_date: dt.date | None = None
    symbol: str | None = Field(default=None, index=True)
    index_id: str | None = Field(default=None, index=True)
    payload_hash: str = Field(index=True)
    raw_payload: str = Field(sa_column=Column(String))
    schema_version: str = Field(default="v1")


class IngestState(SQLModel, table=True):
    provider: str = Field(primary_key=True)
    channel: str = Field(primary_key=True)
    symbol: str = Field(primary_key=True)
    last_ts: dt.datetime | None = None
    last_cursor: str | None = None
    updated_at: dt.datetime = Field(default_factory=utcnow)


class QuoteL2(SQLModel, table=True):
    __table_args__ = (Index("ix_quotes_symbol_ts", "symbol", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    bids: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    asks: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = Field(default="ssi_fcdata")


class TradeTape(SQLModel, table=True):
    __table_args__ = (Index("ix_trades_symbol_ts", "symbol", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    last_price: float
    last_vol: float
    side: str | None = None
    source: str = Field(default="ssi_fcdata")


class ForeignRoom(SQLModel, table=True):
    __table_args__ = (Index("ix_foreign_symbol_ts", "symbol", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    total_room: float | None = None
    current_room: float | None = None
    fbuy_vol: float | None = None
    fsell_vol: float | None = None
    fbuy_val: float | None = None
    fsell_val: float | None = None
    source: str = Field(default="ssi_fcdata")


class IndexOHLCV(SQLModel, table=True):
    __table_args__ = (
        Index("ix_index_ohlcv_id_tf_ts", "index_id", "timeframe", "timestamp", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    index_id: str = Field(index=True)
    timeframe: str = Field(default="1D", index=True)
    timestamp: dt.datetime = Field(index=True)
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    value: float | None = None
    volume: float | None = None
    source: str = Field(default="ssi_fcdata")


class MarketDailyMeta(SQLModel, table=True):
    __table_args__ = (Index("ix_market_daily_meta_symbol_ts", "symbol", "timestamp", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(index=True)
    ref_price: float | None = None
    ceiling_price: float | None = None
    floor_price: float | None = None
    foreign_buy_volume: float | None = None
    foreign_sell_volume: float | None = None
    foreign_buy_value: float | None = None
    foreign_sell_value: float | None = None
    net_foreign_volume: float | None = None
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
    period_end: dt.date | None = None
    public_date: dt.date | None = None
    public_date_is_assumed: bool = False
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

    nim: float | None = None
    casa: float | None = None
    cir: float | None = None
    npl_ratio: float | None = None
    llr_coverage: float | None = None
    credit_growth: float | None = None
    car: float | None = None
    margin_lending_vnd: float | None = None
    adtv_sensitivity: float | None = None
    proprietary_gains_ratio: float | None = None


class CorporateAction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str
    action_type: str
    ex_date: dt.date | None = None
    record_date: dt.date | None = None
    pay_date: dt.date | None = None
    amount: float | None = None
    adjust_method: str = "none"
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class ScreenResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
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
    id: int | None = Field(default=None, primary_key=True)
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
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: dt.datetime = Field(default_factory=utcnow)


class Trade(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
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


class JobRun(SQLModel, table=True):
    __table_args__ = (Index("ix_job_run_job_start", "job_name", "start_ts"),)

    id: int | None = Field(default=None, primary_key=True)
    job_name: str = Field(index=True)
    start_ts: dt.datetime = Field(default_factory=utcnow, index=True)
    end_ts: dt.datetime | None = None
    status: str = Field(default="started", index=True)
    params_hash: str = Field(default="")
    rows_in: int = 0
    rows_out: int = 0


class DataQualityMetric(SQLModel, table=True):
    __table_args__ = (Index("ix_dq_date_provider", "metric_date", "provider"),)

    id: int | None = Field(default=None, primary_key=True)
    metric_date: dt.date = Field(index=True)
    provider: str = Field(index=True)
    symbol: str | None = Field(default=None, index=True)
    timeframe: str | None = Field(default=None, index=True)
    metric_name: str = Field(index=True)
    metric_value: float


class DriftMetric(SQLModel, table=True):
    __table_args__ = (Index("ix_drift_date_metric", "metric_date", "metric_name"),)

    id: int | None = Field(default=None, primary_key=True)
    metric_date: dt.date = Field(index=True)
    metric_name: str = Field(index=True)
    metric_value: float
    alert: bool = False


class BronzeFile(SQLModel, table=True):
    __tablename__ = "bronze_files"
    __table_args__ = (Index("ix_bronze_files_provider_channel_date", "provider", "channel", "date"),)

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    channel: str = Field(index=True)
    date: dt.date = Field(index=True)
    filepath: str
    rows: int = 0
    sha256: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class LastProcessed(SQLModel, table=True):
    __tablename__ = "last_processed"
    __table_args__ = (
        Index(
            "ix_last_processed_scope",
            "provider",
            "channel",
            "symbol",
            "timeframe",
            unique=True,
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    channel: str = Field(index=True)
    symbol: str = Field(default="", index=True)
    timeframe: str = Field(default="", index=True)
    last_ts_utc: dt.datetime | None = None
    updated_at: dt.datetime = Field(default_factory=utcnow)


class MlFeature(SQLModel, table=True):
    __tablename__ = "ml_features"
    __table_args__ = (
        Index("ix_ml_features_symbol_asof", "symbol", "as_of_date"),
        Index("ux_ml_features", "symbol", "as_of_date", "feature_version", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    as_of_date: dt.date = Field(index=True)
    feature_version: str = Field(default="v1", index=True)
    features_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class MlPrediction(SQLModel, table=True):
    __tablename__ = "ml_predictions"
    __table_args__ = (
        Index("ux_ml_predictions", "model_id", "symbol", "as_of_date", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    symbol: str = Field(index=True)
    as_of_date: dt.date = Field(index=True)
    y_hat: float
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class BacktestRun(SQLModel, table=True):
    __tablename__ = "backtest_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_hash: str = Field(index=True)
    config_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    summary_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class BacktestEquityCurve(SQLModel, table=True):
    __tablename__ = "backtest_equity_curve"
    __table_args__ = (Index("ix_backtest_equity_run_date", "run_id", "date", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    date: dt.date = Field(index=True)
    equity: float


class DiagnosticsRun(SQLModel, table=True):
    __tablename__ = "diagnostics_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    model_id: str = Field(index=True)
    config_hash: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class DiagnosticsMetric(SQLModel, table=True):
    __tablename__ = "diagnostics_metrics"
    __table_args__ = (Index("ix_diag_metric_run_name", "run_id", "metric_name"),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    metric_name: str = Field(index=True)
    metric_value: float
    metric_json: JsonDict | None = Field(default=None, sa_column=Column(JSON, nullable=True))

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, BigInteger, Column, DateTime, Index, String
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


class TickerLifecycle(SQLModel, table=True):
    __tablename__ = "ticker_lifecycle"

    symbol: str = Field(primary_key=True)
    first_trading_date: dt.date = Field(index=True)
    last_trading_date: dt.date | None = Field(default=None, index=True)
    exchange: str = Field(index=True)
    sectype: str = Field(index=True)
    sector: str = Field(index=True)
    source: str = Field(default="unknown", index=True)
    updated_at: dt.datetime = Field(default_factory=utcnow)


class IndexMembership(SQLModel, table=True):
    __tablename__ = "index_membership"
    __table_args__ = (
        Index("ix_index_membership_index_date", "index_code", "start_date", "end_date"),
    )

    id: int | None = Field(default=None, primary_key=True)
    index_code: str = Field(index=True)
    symbol: str = Field(index=True)
    start_date: dt.date = Field(index=True)
    end_date: dt.date | None = Field(default=None, index=True)
    source: str = Field(default="unknown", index=True)


class UniverseAudit(SQLModel, table=True):
    __tablename__ = "universe_audit"
    __table_args__ = (Index("ix_universe_audit_date_name", "date", "universe_name", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    date: dt.date = Field(index=True)
    universe_name: str = Field(index=True)
    included_count: int = 0
    excluded_json_breakdown: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class PriceOHLCV(SQLModel, table=True):
    __tablename__ = "prices_ohlcv"
    __table_args__ = (
        Index("ix_prices_timeframe_timestamp", "timeframe", "ts_utc"),
        Index("ix_prices_timestamp", "ts_utc"),
        Index("ix_prices_ohlcv_symbol_timeframe_ts_utc", "symbol", "timeframe", "ts_utc"),
    )

    symbol: str = Field(primary_key=True, index=True)
    timeframe: str = Field(primary_key=True, index=True)
    timestamp: dt.datetime = Field(
        sa_column=Column("ts_utc", DateTime(timezone=False), primary_key=True, nullable=False)
    )

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
    __tablename__ = "quotes_l2"
    __table_args__ = (
        Index("ix_quotes_l2_symbol_ts_utc", "symbol", "ts_utc"),
        Index("ux_quotes_l2_symbol_ts_source", "symbol", "ts_utc", "source", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(
        sa_column=Column("ts_utc", DateTime(timezone=False), nullable=False, index=True)
    )
    bids: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    asks: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = Field(default="ssi_fcdata")


class TradeTape(SQLModel, table=True):
    __tablename__ = "trades_tape"
    __table_args__ = (
        Index("ix_trades_tape_symbol_ts_utc", "symbol", "ts_utc"),
        Index("ux_trades_tape_symbol_ts_source", "symbol", "ts_utc", "source", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: dt.datetime = Field(
        sa_column=Column("ts_utc", DateTime(timezone=False), nullable=False, index=True)
    )
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
    current_room: float | None = None
    total_room: float | None = None
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
    __table_args__ = (
        Index("ix_ca_symbol_exdate", "symbol", "ex_date"),
        Index("ix_ca_symbol_public_date", "symbol", "public_date"),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    action_type: str = Field(index=True)
    ex_date: dt.date
    record_date: dt.date | None = None
    pay_date: dt.date | None = None
    public_date: dt.date | None = None
    params_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = "unknown"
    raw_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class CorporateActionLedger(SQLModel, table=True):
    __tablename__ = "ca_ledger"
    __table_args__ = (
        Index(
            "ux_ca_ledger_pf_symbol_ex_type",
            "portfolio_id",
            "symbol",
            "ex_date",
            "action_type",
            unique=True,
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    portfolio_id: int = Field(index=True)
    symbol: str = Field(index=True)
    ex_date: dt.date = Field(index=True)
    action_type: str
    qty_before: float = 0.0
    qty_after: float = 0.0
    cash_delta: float = 0.0
    avg_cost_before: float = 0.0
    avg_cost_after: float = 0.0
    fee: float = 0.0
    tax: float = 0.0
    notes_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


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
    __tablename__ = "data_quality_metrics"
    __table_args__ = (Index("ix_dq_date_provider", "metric_date", "provider"),)

    id: int | None = Field(default=None, primary_key=True)
    metric_date: dt.date = Field(index=True)
    provider: str = Field(index=True)
    symbol: str | None = Field(default=None, index=True)
    timeframe: str | None = Field(default=None, index=True)
    metric_name: str = Field(index=True)
    metric_value: float


class DriftMetric(SQLModel, table=True):
    __tablename__ = "drift_metrics"
    __table_args__ = (Index("ix_drift_date_metric", "metric_date", "metric_name"),)

    id: int | None = Field(default=None, primary_key=True)
    metric_date: dt.date = Field(index=True)
    metric_name: str = Field(index=True)
    metric_value: float
    alert: bool = False


class DriftAlert(SQLModel, table=True):
    __tablename__ = "drift_alerts"
    __table_args__ = (Index("ix_drift_alert_date_metric", "metric_date", "metric_name"),)

    id: int | None = Field(default=None, primary_key=True)
    metric_date: dt.date = Field(index=True)
    metric_name: str = Field(index=True)
    psi_value: float
    threshold: float = 0.25
    message: str


class BronzeFile(SQLModel, table=True):
    __tablename__ = "bronze_files"
    __table_args__ = (
        Index("ix_bronze_files_provider_channel_date", "provider", "channel", "date"),
        Index(
            "ux_bronze_files_provider_channel_date_hour",
            "provider",
            "channel",
            "date",
            "hour",
            unique=True,
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    channel: str = Field(index=True)
    date: dt.date = Field(index=True)
    hour: int = Field(index=True)
    filepath: str
    rows: int = 0
    sha256: str = Field(index=True)
    created_at_utc: dt.datetime = Field(default_factory=utcnow)


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


class StreamDedup(SQLModel, table=True):
    __tablename__ = "stream_dedup"
    __table_args__ = (
        Index(
            "ux_stream_dedup_provider_rtype_hash", "provider", "rtype", "payload_hash", unique=True
        ),
        Index("ix_stream_dedup_first_seen_at", "first_seen_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    rtype: str = Field(index=True)
    payload_hash: str = Field(index=True)
    first_seen_at: dt.datetime = Field(default_factory=utcnow)


class EventLog(SQLModel, table=True):
    __tablename__ = "event_log"
    __table_args__ = (
        Index("ix_event_log_ts_utc", "ts_utc"),
        Index("ix_event_log_source_type", "source", "event_type"),
        Index("ix_event_log_symbol_ts", "symbol", "ts_utc"),
        Index("ix_event_log_run_id", "run_id"),
        Index("ix_event_log_payload_hash", "payload_hash"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ts_utc: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), nullable=False))
    source: str
    event_type: str
    symbol: str | None = None
    payload_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    payload_hash: str
    run_id: str | None = None


class DailyFlowFeature(SQLModel, table=True):
    __tablename__ = "daily_flow_features"
    __table_args__ = (
        Index("ix_daily_flow_symbol_date", "symbol", "date"),
        Index("ux_daily_flow_symbol_date_source", "symbol", "date", "source", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: dt.date = Field(index=True)
    source: str = Field(default="derived", index=True)
    net_foreign_val_day: float = 0.0
    net_foreign_val_5d: float = 0.0
    net_foreign_val_20d: float = 0.0
    foreign_flow_intensity: float = 0.0
    foreign_room_util: float | None = None


class DailyOrderbookFeature(SQLModel, table=True):
    __tablename__ = "daily_orderbook_features"
    __table_args__ = (
        Index("ix_daily_orderbook_symbol_date", "symbol", "date"),
        Index("ux_daily_orderbook_symbol_date_source", "symbol", "date", "source", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: dt.date = Field(index=True)
    source: str = Field(default="derived", index=True)
    imb_1_day: float = 0.0
    imb_3_day: float = 0.0
    spread_day: float = 0.0


class DailyIntradayFeature(SQLModel, table=True):
    __tablename__ = "daily_intraday_features"
    __table_args__ = (
        Index("ix_daily_intraday_symbol_date", "symbol", "date"),
        Index("ux_daily_intraday_symbol_date_source", "symbol", "date", "source", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: dt.date = Field(index=True)
    source: str = Field(default="derived", index=True)
    rv_day: float = 0.0
    vol_first_hour_ratio: float = 0.0


class FeatureLastProcessed(SQLModel, table=True):
    __tablename__ = "feature_last_processed"
    __table_args__ = (Index("ux_feature_last_processed", "feature_name", "symbol", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    feature_name: str = Field(index=True)
    symbol: str = Field(default="", index=True)
    last_date: dt.date
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
    feature_version: str = Field(default="v3", index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)
    data_coverage_score: float = Field(default=0.0)

    ret_1d: float | None = None
    ret_5d: float | None = None
    ret_21d: float | None = None
    ret_63d: float | None = None
    ret_126d: float | None = None
    ret_252d: float | None = None
    rev_5d: float | None = None
    vol_20d: float | None = None
    vol_60d: float | None = None
    vol_120d: float | None = None
    atr14_pct: float | None = None
    adv20_value: float | None = None
    adv20_vol: float | None = None
    spread_proxy: float | None = None
    limit_hit_20d: float | None = None
    rsi14: float | None = None
    macd_hist: float | None = None
    ema20_gt_ema50: float | None = None
    close_gt_ema50: float | None = None
    ema50_slope: float | None = None
    value_score_z: float | None = None
    quality_score_z: float | None = None
    momentum_score_z: float | None = None
    lowvol_score_z: float | None = None
    dividend_score_z: float | None = None
    regime_trend_up: float | None = None
    regime_sideways: float | None = None
    regime_risk_off: float | None = None
    net_foreign_val_5d: float | None = None
    net_foreign_val_20d: float | None = None
    foreign_flow_intensity: float | None = None
    foreign_room_util: float | None = None
    imb_1_day: float | None = None
    imb_3_day: float | None = None
    spread_day: float | None = None
    rv_day: float | None = None
    vol_first_hour_ratio: float | None = None
    fundamental_public_date_is_assumed: float | None = None
    fundamental_public_date_limitation_flag: float | None = None
    y_excess: float | None = None
    y_rank_z: float | None = None


class FeatureCoverage(SQLModel, table=True):
    __tablename__ = "feature_coverage"
    __table_args__ = (Index("ux_feature_coverage", "date", "feature_version", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    date: dt.date = Field(index=True)
    feature_version: str = Field(default="v3", index=True)
    metrics_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class ConformalState(SQLModel, table=True):
    __tablename__ = "conformal_state"
    __table_args__ = (Index("ux_conformal_state", "model_id", "bucket_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    bucket_id: int = Field(index=True)
    alpha_b: float = 0.20
    miss_ema: float = 0.20
    updated_at: dt.datetime = Field(default_factory=utcnow)


class ConformalResidual(SQLModel, table=True):
    __tablename__ = "conformal_residuals"
    __table_args__ = (Index("ux_conformal_residual", "model_id", "date", "symbol", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    date: dt.date = Field(index=True)
    symbol: str = Field(index=True)
    bucket_id: int = Field(index=True)
    abs_residual: float
    miss: float
    created_at: dt.datetime = Field(default_factory=utcnow)


class ConformalBucketSpec(SQLModel, table=True):
    __tablename__ = "conformal_bucket_spec"
    __table_args__ = (Index("ux_conformal_bucket_spec", "model_id", "month_start", "bucket_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    month_start: dt.date = Field(index=True)
    bucket_id: int = Field(index=True)
    low: float | None = None
    high: float | None = None
    created_at: dt.datetime = Field(default_factory=utcnow)


class ConformalCoverageDaily(SQLModel, table=True):
    __tablename__ = "conformal_coverage_daily"
    __table_args__ = (Index("ux_conformal_coverage_daily", "model_id", "date", "bucket_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    date: dt.date = Field(index=True)
    bucket_id: int = Field(index=True)
    coverage: float
    interval_half_width: float
    count: int
    created_at: dt.datetime = Field(default_factory=utcnow)




class MlLabel(SQLModel, table=True):
    __tablename__ = "ml_labels"
    __table_args__ = (
        Index("ix_ml_labels_symbol_date", "symbol", "date"),
        Index("ux_ml_labels", "symbol", "date", "label_version", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: dt.date = Field(index=True)
    y_excess: float
    y_rank_z: float
    label_version: str = Field(default="v3", index=True)


class MlPrediction(SQLModel, table=True):
    __tablename__ = "ml_predictions"
    __table_args__ = (Index("ux_ml_predictions", "model_id", "symbol", "as_of_date", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    symbol: str = Field(index=True)
    as_of_date: dt.date = Field(index=True)
    y_hat: float
    meta: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))


class AlphaModel(SQLModel, table=True):
    __tablename__ = "alpha_models"
    __table_args__ = (Index("ux_alpha_models", "model_id", "version", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    version: str = Field(index=True)
    artifact_path: str
    train_start: dt.date
    train_end: dt.date
    config_hash: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class AlphaPrediction(SQLModel, table=True):
    __tablename__ = "alpha_predictions"
    __table_args__ = (
        Index("ux_alpha_predictions", "model_id", "as_of_date", "symbol", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    as_of_date: dt.date = Field(index=True)
    symbol: str = Field(index=True)
    score: float
    mu: float
    uncert: float
    pred_base: float
    created_at: dt.datetime = Field(default_factory=utcnow)


class BacktestRun(SQLModel, table=True):
    __tablename__ = "backtest_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_hash: str = Field(index=True)
    config_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    summary_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class BacktestMetric(SQLModel, table=True):
    __tablename__ = "backtest_metrics"
    __table_args__ = (Index("ix_backtest_metrics_run_metric", "run_id", "metric_name", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    metric_name: str = Field(index=True)
    metric_value: float


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


class DsrResult(SQLModel, table=True):
    __tablename__ = "dsr_results"
    __table_args__ = (Index("ux_dsr_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    dsr_value: float
    components: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class PboResult(SQLModel, table=True):
    __tablename__ = "pbo_results"
    __table_args__ = (Index("ux_pbo_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    phi: float
    logits_summary: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class PsrResult(SQLModel, table=True):
    __tablename__ = "psr_results"
    __table_args__ = (Index("ux_psr_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    psr_value: float
    sr_hat: float
    sr_threshold: float
    t: int
    skew: float
    kurt: float
    created_at: dt.datetime = Field(default_factory=utcnow)


class MinTrlResult(SQLModel, table=True):
    __tablename__ = "mintrl_results"
    __table_args__ = (Index("ux_mintrl_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    mintrl: int
    sr_hat: float
    sr_threshold: float
    alpha: float
    created_at: dt.datetime = Field(default_factory=utcnow)


class RealityCheckResult(SQLModel, table=True):
    __tablename__ = "reality_check_results"
    __table_args__ = (Index("ux_reality_check_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    p_value: float
    components: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class SpaResult(SQLModel, table=True):
    __tablename__ = "spa_results"
    __table_args__ = (Index("ux_spa_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    p_value: float
    components: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class GateResult(SQLModel, table=True):
    __tablename__ = "gate_results"
    __table_args__ = (Index("ux_gate_results_run_id", "run_id", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    status: str = Field(index=True)
    reasons: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    details: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class ParquetManifest(SQLModel, table=True):
    __tablename__ = "parquet_manifest"
    __table_args__ = (
        Index("ix_parquet_manifest_dataset_day", "dataset", "year", "month", "day"),
        Index("ux_parquet_manifest_dataset_partition", "dataset", "year", "month", "day", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    dataset: str = Field(index=True)
    year: int = Field(index=True)
    month: int = Field(index=True)
    day: int = Field(index=True)
    file_path: str
    row_count: int
    schema_hash: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class RebalanceConstraintReport(SQLModel, table=True):
    __tablename__ = "rebalance_constraint_reports"
    __table_args__ = (Index("ix_rebalance_constraint_reports_date_tag", "as_of_date", "run_tag"),)

    id: int | None = Field(default=None, primary_key=True)
    as_of_date: dt.date = Field(index=True)
    run_tag: str = Field(index=True, default="alpha_v3")
    report_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)

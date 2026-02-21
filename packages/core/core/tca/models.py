from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Column, DateTime, Index
from sqlmodel import Field, SQLModel


class OrderTCA(SQLModel, table=True):
    __tablename__ = "order_tca"
    __table_args__ = (
        Index("ix_order_tca_order", "order_id", unique=True),
        Index("ix_order_tca_created", "created_at"),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    order_id: str = Field(index=True)
    market: str = Field(index=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(default="1D")
    arrival_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    arrival_price: float
    exec_start_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    exec_end_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    exec_vwap_price: float
    benchmark_twap_price: float
    benchmark_vwap_price: float
    side: str = Field(index=True)
    qty_requested: float
    qty_filled: float
    notional_arrival: float
    fee_total: float = 0.0
    tax_total: float = 0.0
    slippage_total: float = 0.0
    funding_total: float = 0.0
    is_price_component: float = 0.0
    is_total: float = 0.0
    is_bps_total: float = 0.0
    spread_bps_est: float = 0.0
    delay_cost: float = 0.0
    impact_cost: float = 0.0
    quality_bucket: str = "Vừa"
    reason_vi: str = "Có trượt giá/chi phí ở mức trung bình."
    config_hash: str = ""
    dataset_hash: str = ""
    code_hash: str = ""
    report_id: str = ""
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class TCABenchmarkPoint(SQLModel, table=True):
    __tablename__ = "tca_benchmark_points"
    __table_args__ = (Index("ix_tca_bench_order_ts", "order_id", "ts"),)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    order_id: str = Field(index=True)
    ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    bar_price_ref: float
    bar_volume: float
    bar_vwap_est: float

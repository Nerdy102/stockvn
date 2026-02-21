from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel


JsonDict = dict[str, Any]


class Order(SQLModel, table=True):
    __tablename__ = "oms_orders"
    __table_args__ = (
        Index("ux_oms_orders_idempotency_key", "idempotency_key", unique=True),
        Index("ix_oms_orders_created_at", "created_at"),
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    market: str = Field(index=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(default="1D")
    mode: str = Field(default="paper", index=True)
    order_type: str = Field(default="market", index=True)
    execution_pref: str = Field(default="close", index=True)
    side: str = Field(index=True)
    qty: float
    price: float | None = None
    status: str = Field(default="DRAFT", index=True)
    idempotency_key: str = Field(index=True)
    client_order_id: str | None = Field(default=None, index=True)
    broker_order_id: str | None = Field(default=None, index=True)
    notional_est: float = 0.0
    fee_est: float = 0.0
    tax_est: float = 0.0
    slippage_est: float = 0.0
    reason_short: str = ""
    risk_tags_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    model_id: str = ""
    config_hash: str = ""
    dataset_hash: str = ""
    code_hash: str = ""
    confirm_token: str | None = Field(default=None, index=True)
    confirm_used_at: dt.datetime | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class OrderEvent(SQLModel, table=True):
    __tablename__ = "oms_order_events"
    __table_args__ = (Index("ix_oms_order_events_order_ts", "order_id", "ts"),)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    order_id: str = Field(index=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    from_status: str
    to_status: str
    event_type: str
    payload_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    correlation_id: str = Field(index=True)


class Fill(SQLModel, table=True):
    __tablename__ = "oms_fills"
    __table_args__ = (Index("ix_oms_fills_order_ts", "order_id", "ts"),)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    order_id: str = Field(index=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    fill_qty: float
    fill_price: float
    fee: float = 0.0
    tax: float = 0.0
    slippage_cost: float = 0.0
    funding_cost: float = 0.0
    pnl_gross: float | None = None
    pnl_net: float | None = None
    broker_fill_id: str | None = Field(default=None, index=True)
    correlation_id: str = Field(index=True)


class PortfolioSnapshot(SQLModel, table=True):
    __tablename__ = "oms_portfolio_snapshots"

    ts: dt.datetime = Field(primary_key=True)
    cash: float
    positions_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    nav_est: float
    drawdown_est: float = 0.0

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


SignalLabel = Literal["TANG", "GIAM", "TRUNG_TINH", "UU_TIEN_QUAN_SAT"]
ConfidenceLabel = Literal["THAP", "VUA", "CAO"]


class SignalResult(BaseModel):
    symbol: str
    timeframe: str
    model_id: str
    as_of: dt.datetime
    signal: SignalLabel
    confidence: ConfidenceLabel
    proposed_side: Literal["BUY", "SELL", "SHORT", "HOLD"] = "HOLD"
    explanation: list[str] = Field(default_factory=list)
    reason_short: str = ""
    reason_bullets: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list, max_length=2)
    confidence_bucket: Literal["Thấp", "Vừa", "Cao"] = "Thấp"
    risks: list[str] = Field(default_factory=list)
    indicators: dict[str, float] = Field(default_factory=dict)
    debug_fields: dict[str, Any] = Field(default_factory=dict)
    latest_price: float = 0.0
    marker_time: str | None = None


class FeeTaxPreview(BaseModel):
    commission: float
    sell_tax: float
    slippage_est: float
    total_cost: float


class OrderDraft(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL", "SHORT"]
    ui_side: Literal["MUA", "BAN", "MO_VI_THE_BAN"]
    qty: int
    price: float
    notional: float
    fee_tax: FeeTaxPreview
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    mode: Literal["paper", "draft", "live"] = "draft"
    off_session: bool = False


class BacktestReport(BaseModel):
    model_id: str
    symbols: list[str]
    start: dt.date
    end: dt.date
    cagr: float
    mdd: float
    sharpe: float
    sortino: float
    turnover: float
    net_return: float
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    config_hash: str
    dataset_hash: str
    code_hash: str


class ConfirmExecutePayload(BaseModel):
    portfolio_id: int = 1
    user_id: str = "anonymous"
    session_id: str = "anonymous-session"
    idempotency_token: str | None = None
    mode: Literal["paper", "draft", "live"] = "paper"
    acknowledged_educational: bool = False
    acknowledged_loss: bool = False
    acknowledged_live_eligibility: bool = False
    age: int | None = None
    draft: OrderDraft

from __future__ import annotations

import datetime as dt

from sqlmodel import Field, SQLModel


class SignalAudit(SQLModel, table=True):
    __tablename__ = "signal_audit"

    id: int | None = Field(default=None, primary_key=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    market: str = Field(index=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)
    model_id: str = Field(index=True)
    signal: str = Field(index=True)
    confidence_bucket: str = Field(index=True)
    confidence_score: int = 0
    reason_short: str = ""
    risk_tags_json: str = "[]"
    debug_fields_json: str = "{}"
    config_hash: str = Field(index=True)
    dataset_hash: str = Field(index=True)
    code_hash: str = Field(index=True)

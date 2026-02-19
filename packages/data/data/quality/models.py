from __future__ import annotations

import datetime as dt

from sqlmodel import Field, SQLModel


class DataQualityEvent(SQLModel, table=True):
    __tablename__ = "data_quality_events"

    id: int | None = Field(default=None, primary_key=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    market: str = Field(index=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)
    severity: str = Field(index=True)
    code: str = Field(index=True)
    message: str
    dataset_hash: str = Field(index=True)

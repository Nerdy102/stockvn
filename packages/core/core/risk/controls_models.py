from __future__ import annotations

import datetime as dt

from sqlmodel import Field, SQLModel


class TradingControl(SQLModel, table=True):
    __tablename__ = "trading_controls"

    id: int = Field(default=1, primary_key=True)
    kill_switch_enabled: bool = Field(default=False, index=True)
    paused_reason_code: str | None = Field(default=None, index=True)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)

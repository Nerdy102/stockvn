from __future__ import annotations

import datetime as dt

from sqlmodel import Field, SQLModel


class CorporateAction(SQLModel, table=True):
    __tablename__ = "corporate_actions_vn"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    ex_date: dt.date = Field(index=True)
    action_type: str = Field(index=True)
    amount: float
    note: str = ""

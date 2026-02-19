from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class ReconcileReport(SQLModel, table=True):
    __tablename__ = "reconcile_reports"

    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, primary_key=True)
    status: str = Field(default="OK", index=True)
    mismatches_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    fixed_actions_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

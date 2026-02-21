from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class CostCalibrationReport(SQLModel, table=True):
    __tablename__ = "cost_calibration_reports"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    params_new_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

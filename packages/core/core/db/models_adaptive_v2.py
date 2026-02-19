from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index
from sqlmodel import Field, SQLModel


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


JsonDict = dict[str, Any]


class ChangePointV2(SQLModel, table=True):
    __tablename__ = "change_points_v2"
    __table_args__ = (Index("ix_change_points_v2_series_tf_ts", "series_key", "tf", "detected_at_ts"),)

    id: int | None = Field(default=None, primary_key=True)
    series_key: str = Field(index=True)
    tf: str = Field(index=True)
    detected_at_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    cp_type: str = Field(index=True)
    stat: float
    threshold: float
    severity: str = Field(index=True)
    window_short: int
    window_long: int
    candidates_checked: int
    metadata_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class RegimeStateV2(SQLModel, table=True):
    __tablename__ = "regime_state_v2"

    id: int | None = Field(default=None, primary_key=True)
    as_of_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    tf: str = Field(index=True)
    probs_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    active_regime: str = Field(index=True)
    hysteresis_applied: bool = False
    features_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=utcnow)


class UncertaintyCalibratorV2(SQLModel, table=True):
    __tablename__ = "uncertainty_calibrators_v2"

    id: int | None = Field(default=None, primary_key=True)
    tf: str = Field(index=True)
    method: str = Field(index=True)
    state: str | None = Field(default=None, index=True)
    alpha_t: float | None = None
    q_t: float | None = None
    q_state0: float | None = None
    q_state1: float | None = None
    cooldown_until_ts: dt.datetime | None = None
    window_stats_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: dt.datetime = Field(default_factory=utcnow)


class ExpertWeightsV2(SQLModel, table=True):
    __tablename__ = "expert_weights_v2"

    id: int | None = Field(default=None, primary_key=True)
    as_of_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    tf: str = Field(default="60m", index=True)
    regime: str = Field(index=True)
    weights_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    rewards_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: dt.datetime = Field(default_factory=utcnow)


class TCAFillV2(SQLModel, table=True):
    __tablename__ = "tca_fills_v2"

    id: int | None = Field(default=None, primary_key=True)
    decision_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    submit_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    fill_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    intended_price: float
    executed_price: float
    notional: float
    participation_rate: float
    slippage_bps: float
    session: str = Field(index=True)
    regime: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class SectorUncertaintyV2(SQLModel, table=True):
    __tablename__ = "sector_uncertainty_v2"

    id: int | None = Field(default=None, primary_key=True)
    sector: str = Field(index=True)
    tf: str = Field(index=True)
    q_sector: float
    stats_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: dt.datetime = Field(default_factory=utcnow)


class AlphaPredictionIntradayV2(SQLModel, table=True):
    __tablename__ = "alpha_predictions_intraday_v2"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    tf: str = Field(index=True)
    end_ts: dt.datetime = Field(sa_column=Column(DateTime(timezone=False), index=True))
    alpha_score: float
    alpha_confidence: float
    regime: str = Field(index=True)
    weights_used_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    prediction_hash: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)


class TCAParamsV2(SQLModel, table=True):
    __tablename__ = "tca_params_v2"

    id: int | None = Field(default=None, primary_key=True)
    bucket_key: str = Field(index=True)
    date: dt.date = Field(index=True)
    params_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    metrics_json: JsonDict = Field(default_factory=dict, sa_column=Column(JSON))
    param_hash: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=utcnow)

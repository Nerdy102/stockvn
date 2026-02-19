from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .canonical import derive_event_id, hash_payload, strict_mapping


def _require_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _require_dt(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    return value


@dataclass(frozen=True)
class FeatureSnapshotV2:
    as_of_ts_utc: datetime
    symbol: str
    timeframe: str
    feature_version: str
    features_json: Mapping[str, Any]
    lineage_json: Mapping[str, Any]
    event_id: str = field(default="", kw_only=True)
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_dt("as_of_ts_utc", self.as_of_ts_utc)
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("timeframe", self.timeframe)
        _require_non_empty("feature_version", self.feature_version)

        features_json = strict_mapping(self.features_json, field_name="features_json")
        lineage_json = strict_mapping(self.lineage_json, field_name="lineage_json")

        missing = {"bar_hashes_used", "config_hash", "code_hash"} - set(lineage_json)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"lineage_json missing required fields: {joined}")

        body = {
            "as_of_ts_utc": self.as_of_ts_utc.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_version": self.feature_version,
            "features_json": features_json,
            "lineage_json": lineage_json,
        }
        event_id = self.event_id or derive_event_id(body)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "feature_hash", hash_payload({**body, "event_id": event_id}))


@dataclass(frozen=True)
class AlphaPredictionV2:
    as_of_ts_utc: datetime
    symbol: str
    horizon: str
    score: float
    model_id: str
    model_hash: str
    calibration_id: str
    uncertainty_id: str
    p_outperform: float | None = None
    interval_lo: float | None = None
    interval_hi: float | None = None
    event_id: str = field(default="", kw_only=True)
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_dt("as_of_ts_utc", self.as_of_ts_utc)
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("horizon", self.horizon)
        _require_non_empty("model_id", self.model_id)
        _require_non_empty("model_hash", self.model_hash)
        _require_non_empty("calibration_id", self.calibration_id)
        _require_non_empty("uncertainty_id", self.uncertainty_id)

        body = {
            "as_of_ts_utc": self.as_of_ts_utc.isoformat(),
            "symbol": self.symbol,
            "horizon": self.horizon,
            "score": float(self.score),
            "p_outperform": None if self.p_outperform is None else float(self.p_outperform),
            "interval_lo": None if self.interval_lo is None else float(self.interval_lo),
            "interval_hi": None if self.interval_hi is None else float(self.interval_hi),
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "calibration_id": self.calibration_id,
            "uncertainty_id": self.uncertainty_id,
        }
        event_id = self.event_id or derive_event_id(body)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "prediction_hash", hash_payload({**body, "event_id": event_id}))

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.features_v2 import AlphaPredictionV2, FeatureSnapshotV2


def test_feature_snapshot_v2_requires_lineage_fields() -> None:
    with pytest.raises(ValueError):
        FeatureSnapshotV2(
            as_of_ts_utc=datetime.now(timezone.utc),
            symbol="AAA",
            timeframe="60m",
            feature_version="v2",
            features_json={"x": 1.0},
            lineage_json={"config_hash": "c", "code_hash": "k"},
        )


def test_alpha_prediction_v2_hash_is_deterministic() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    p1 = AlphaPredictionV2(
        as_of_ts_utc=ts,
        symbol="AAA",
        horizon="1b",
        score=0.2,
        model_id="m",
        model_hash="h",
        calibration_id="c",
        uncertainty_id="u",
    )
    p2 = AlphaPredictionV2(
        as_of_ts_utc=ts,
        symbol="AAA",
        horizon="1b",
        score=0.2,
        model_id="m",
        model_hash="h",
        calibration_id="c",
        uncertainty_id="u",
    )
    assert p1.prediction_hash == p2.prediction_hash

from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_listnet_v2_reliability_api_shape() -> None:
    c = TestClient(create_app())
    r = c.get("/ml/listnet_v2/reliability")
    assert r.status_code == 200
    body = r.json()
    assert "prob_calibration_metrics" in body
    assert "governance_warning" in body
    assert "rolling_ece_20" in body

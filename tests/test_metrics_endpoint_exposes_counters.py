from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_metrics_endpoint_exposes_prometheus_counters() -> None:
    c = TestClient(create_app())
    r = c.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "ingest_rows_total" in text
    assert "ingest_errors_total" in text
    assert "upsert_rows_total" in text

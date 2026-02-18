from __future__ import annotations

import json
from pathlib import Path

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_dashboard_contains_all_sections() -> None:
    golden = json.loads(Path("tests/golden/portfolio_dashboard_schema_keys.json").read_text())
    c = TestClient(create_app())
    r = c.get("/portfolio/dashboard")
    assert r.status_code == 200
    body = r.json()
    for key in golden["required_top_level"]:
        assert key in body
    for key in golden["required_risk_keys"]:
        assert key in body["risk"]
    for key in golden["required_constraints_keys"]:
        assert key in body["constraints"]
    for key in golden["required_capacity_keys"]:
        assert key in body["capacity"]

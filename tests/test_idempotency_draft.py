from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_idempotency_draft() -> None:
    client = TestClient(app)
    uid = uuid.uuid4().hex[:8]
    payload = {
        "user_id": "u-oms",
        "market": "vn",
        "symbol": "FPT",
        "side": "BUY",
        "qty": 100,
        "price": 10000,
        "model_id": "m1",
        "config_hash": "cfg-a",
        "ts_bucket": uid,
    }
    r1 = client.post("/oms/draft", json=payload)
    r2 = client.post("/oms/draft", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["order"]["id"] == r2.json()["order"]["id"]

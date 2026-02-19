from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_audit_log_append_only() -> None:
    client = TestClient(app)
    before = client.get('/oms/audit', params={'limit': 500}).json()['items']

    draft = client.post(
        "/oms/draft",
        json={
            "user_id": "u-oms",
            "market": "vn",
            "symbol": "FPT",
            "side": "BUY",
            "qty": 100,
            "price": 10000,
            "model_id": "m1",
            "config_hash": "cfg-audit",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]
    client.post('/oms/approve', json={'order_id': draft['id'], 'confirm_token': draft['confirm_token'], 'checkboxes': {'a': True}})

    after = client.get('/oms/audit', params={'limit': 500}).json()['items']
    assert len(after) >= len(before) + 1

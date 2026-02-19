from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_confirm_token_single_use() -> None:
    client = TestClient(app)
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
            "config_hash": "cfg-single",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]

    payload = {
        "order_id": draft["id"],
        "confirm_token": draft["confirm_token"],
        "checkboxes": {"xac_nhan_rui_ro": True, "xac_nhan_khong_tu_dong": True},
    }
    ok = client.post("/oms/approve", json=payload)
    assert ok.status_code == 200
    blocked = client.post("/oms/approve", json=payload)
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["reason_code"] == "CONFIRM_ALREADY_USED"

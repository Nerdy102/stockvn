from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_no_live_when_disabled() -> None:
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
            "mode": "live",
            "model_id": "m1",
            "config_hash": "cfg-live-disabled",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]
    client.post(
        "/oms/approve",
        json={"order_id": draft["id"], "confirm_token": draft["confirm_token"], "checkboxes": {"a": True}},
    )
    blocked = client.post(
        "/oms/execute",
        json={
            "order_id": draft["id"],
            "data_freshness": {"as_of_date": dt.date.today().isoformat()},
            "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
            "user_age": 30,
        },
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["reason_code"] == "LIVE_DISABLED"

from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_kill_switch_blocks_execute() -> None:
    c = TestClient(app)
    draft = c.post(
        "/oms/draft",
        json={
            "user_id": "u1",
            "market": "vn",
            "symbol": "FPT",
            "side": "BUY",
            "qty": 100,
            "price": 10000,
            "config_hash": f"cfg-{uuid.uuid4().hex[:6]}",
            "model_id": "m1",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]
    c.post("/oms/approve", json={"order_id": draft["id"], "confirm_token": draft["confirm_token"], "checkboxes": {"ok": True}})
    c.post("/controls/kill_switch/on", json={})

    blocked = c.post(
        "/oms/execute",
        json={
            "order_id": draft["id"],
            "data_freshness": {"as_of_date": dt.date.today().isoformat()},
            "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
        },
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["reason_code"] == "KILL_SWITCH_ON"
    c.post("/controls/kill_switch/off", json={})

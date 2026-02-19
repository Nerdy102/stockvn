from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_state_machine_transitions() -> None:
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
            "config_hash": "cfg-sm",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]
    client.post(
        "/oms/approve",
        json={
            "order_id": draft["id"],
            "confirm_token": draft["confirm_token"],
            "checkboxes": {"a": True, "b": True},
        },
    )

    res = client.post(
        "/oms/execute",
        json={
            "order_id": draft["id"],
            "data_freshness": {"as_of_date": dt.date.today().isoformat()},
            "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
            "drift_alerts": {"drift_paused": False, "kill_switch_on": False},
        },
    )
    assert res.status_code == 200
    assert res.json()["order"]["status"] == "FILLED"

    audit = client.get("/oms/audit", params={"limit": 200}).json()["items"]
    order_events = [x for x in audit if x["order_id"] == draft["id"]]
    pairs = {(x["from_status"], x["to_status"]) for x in order_events}
    assert ("DRAFT", "APPROVED") in pairs
    assert ("APPROVED", "SENT") in pairs
    assert ("SENT", "ACKED") in pairs
    assert ("ACKED", "FILLED") in pairs

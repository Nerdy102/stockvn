from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def _create_and_approve(client: TestClient, market: str = "vn") -> dict:
    draft = client.post(
        "/oms/draft",
        json={
            "user_id": "u-oms",
            "market": market,
            "symbol": "FPT" if market == "vn" else "BTCUSDT",
            "side": "BUY",
            "qty": 100,
            "price": 10000,
            "model_id": "m1",
            "config_hash": f"cfg-risk-{uuid.uuid4().hex[:8]}",
            "ts_bucket": uuid.uuid4().hex[:8],
        },
    ).json()["order"]
    client.post(
        "/oms/approve",
        json={"order_id": draft["id"], "confirm_token": draft["confirm_token"], "checkboxes": {"a": True}},
    )
    return draft


def test_risk_blocks_execute() -> None:
    client = TestClient(app)
    draft1 = _create_and_approve(client, "vn")
    stale = client.post(
        "/oms/execute",
        json={
            "order_id": draft1["id"],
            "data_freshness": {"as_of_date": (dt.date.today() - dt.timedelta(days=3)).isoformat()},
            "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
        },
    )
    assert stale.status_code == 403
    assert stale.json()["detail"]["reason_code"] == "DATA_STALE"

    draft2 = _create_and_approve(client, "vn")
    off = client.post(
        "/oms/execute",
        json={
            "order_id": draft2["id"],
            "outside_session_vn": True,
            "data_freshness": {"as_of_date": dt.date.today().isoformat()},
            "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
            "drift_alerts": {"drift_paused": False, "kill_switch_on": False},
        },
    )
    assert off.status_code == 403
    assert off.json()["detail"]["reason_code"] == "SESSION_OFF_HOURS"

    draft3 = _create_and_approve(client, "vn")
    cash = client.post(
        "/oms/execute",
        json={
            "order_id": draft3["id"],
            "data_freshness": {"as_of_date": dt.date.today().isoformat()},
            "portfolio_snapshot": {"cash": 1_000.0, "nav_est": 1_000_000_000.0, "orders_today": 0},
        },
    )
    assert cash.status_code == 403
    assert cash.json()["detail"]["reason_code"] == "RISK_CASH_BUFFER"

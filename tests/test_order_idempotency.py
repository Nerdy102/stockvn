from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_order_submit_idempotency() -> None:
    c = TestClient(create_app())
    payload = {
        "portfolio_id": 1,
        "client_order_id": "idem-001",
        "symbol": "AAA",
        "side": "BUY",
        "quantity": 100,
        "price": 10000,
        "adapter": "paper",
    }
    r1 = c.post("/orders/submit", json=payload)
    assert r1.status_code == 200
    r2 = c.post("/orders/submit", json=payload)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["idempotent"] is True

    rows = c.get("/orders", params={"limit": 500}).json()
    assert len([x for x in rows if x["client_order_id"] == "idem-001"]) == 1

from __future__ import annotations

from core.brokers.sandbox import LiveBrokerSandbox


def test_broker_sandbox_place_ack_fill_flow() -> None:
    broker = LiveBrokerSandbox()
    ack = broker.place_order({"symbol": "BTCUSDT", "side": "BUY", "qty": 0.01, "price": 50000})
    assert ack["ack"] is True
    oid = ack["broker_order_id"]

    status = broker.get_order_status(oid)
    assert status["status"] == "FILLED"

    fills = broker.get_fills()
    assert any(f["broker_order_id"] == oid for f in fills)

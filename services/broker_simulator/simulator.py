from __future__ import annotations

import datetime as dt
import hashlib

from core.brokers.base import BrokerAck


class BrokerSimulator:
    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.fills: list[dict] = []

    def place(self, order: dict) -> BrokerAck:
        seed = f"{order.get('idempotency_key')}|{order.get('symbol')}|{order.get('qty')}|{order.get('price')}"
        broker_order_id = hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]
        if broker_order_id not in self.orders:
            self.orders[broker_order_id] = {
                "broker_order_id": broker_order_id,
                "status": "ACKED",
                "created_at": dt.datetime.utcnow().isoformat(),
            }
            self.fills.append(
                {
                    "broker_order_id": broker_order_id,
                    "broker_fill_id": f"fill-{broker_order_id}",
                    "fill_qty": float(order.get("qty", 0.0)),
                    "fill_price": float(order.get("price") or 0.0),
                    "ts": dt.datetime.utcnow().isoformat(),
                }
            )
        return BrokerAck(broker_order_id=broker_order_id, status="ACKED", message="Mô phỏng sandbox ACK")

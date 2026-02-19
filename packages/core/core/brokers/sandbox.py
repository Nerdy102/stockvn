from __future__ import annotations

import datetime as dt
import hashlib


class LiveBrokerSandbox:
    """Broker sandbox deterministic cho kiểm thử e2e place->ack->fill."""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._fills: list[dict] = []

    def place_order(self, order: dict) -> dict:
        raw = f"{order.get('symbol')}|{order.get('side')}|{order.get('qty')}|{order.get('price')}"
        broker_order_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        if broker_order_id not in self._orders:
            self._orders[broker_order_id] = {
                "broker_order_id": broker_order_id,
                "status": "ACKED",
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "qty": float(order.get("qty", 0)),
                "price": float(order.get("price", 0)),
                "created_at": dt.datetime.utcnow().isoformat(),
            }
            self._fills.append(
                {
                    "broker_order_id": broker_order_id,
                    "execution_id": f"fill-{broker_order_id}",
                    "qty": float(order.get("qty", 0)),
                    "price": float(order.get("price", 0)),
                    "filled_at": dt.datetime.utcnow().isoformat(),
                }
            )
        return {"broker_order_id": broker_order_id, "ack": True}

    def cancel_order(self, broker_order_id: str) -> dict:
        row = self._orders.get(broker_order_id)
        if row is None:
            return {"ok": False, "reason": "not_found"}
        row["status"] = "CANCELLED"
        return {"ok": True}

    def get_order_status(self, broker_order_id: str) -> dict:
        row = self._orders.get(broker_order_id)
        if row is None:
            return {"status": "UNKNOWN"}
        if row["status"] == "ACKED":
            return {"status": "FILLED", "broker_order_id": broker_order_id}
        return {"status": row["status"], "broker_order_id": broker_order_id}

    def get_fills(self, since: str | None = None) -> list[dict]:
        del since
        return list(self._fills)

    def get_balances(self) -> dict:
        return {"cash": 1_000_000_000.0}

    def get_positions(self) -> list[dict]:
        return []

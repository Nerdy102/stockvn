from __future__ import annotations

from core.brokers.base import BrokerAck
from services.broker_simulator.simulator import BrokerSimulator


class SandboxBroker:
    def __init__(self, simulator: BrokerSimulator | None = None) -> None:
        self._sim = simulator or BrokerSimulator()

    def place_order(self, order: dict) -> BrokerAck:
        return self._sim.place(order)

    def cancel_order(self, broker_order_id: str) -> dict:
        row = self._sim.orders.get(broker_order_id)
        if row is None:
            return {"ok": False, "reason": "not_found"}
        row["status"] = "CANCELLED"
        return {"ok": True}

    def get_order(self, broker_order_id: str) -> dict:
        return self._sim.orders.get(broker_order_id, {"status": "UNKNOWN"})

    def get_order_status(self, broker_order_id: str) -> dict:
        row = self.get_order(broker_order_id)
        if any(f.get("broker_order_id") == broker_order_id for f in self._sim.fills):
            row = dict(row)
            row["status"] = "FILLED"
        return row

    def get_fills(self, since_ts: str | None = None) -> list[dict]:
        del since_ts
        return list(self._sim.fills)

    def get_balances(self) -> dict:
        return {"cash": 1_000_000_000.0}

    def get_positions(self) -> list[dict]:
        return []


class LiveBrokerSandbox(SandboxBroker):
    """Tương thích ngược với tên cũ."""

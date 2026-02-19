from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BrokerAck:
    broker_order_id: str
    status: str
    message: str = ""


class BrokerAdapter(Protocol):
    def place_order(self, order: dict) -> BrokerAck: ...

    def cancel_order(self, broker_order_id: str) -> dict: ...

    def get_order(self, broker_order_id: str) -> dict: ...

    def get_fills(self, since_ts: str | None = None) -> list[dict]: ...

    def get_balances(self) -> dict: ...

    def get_positions(self) -> list[dict]: ...

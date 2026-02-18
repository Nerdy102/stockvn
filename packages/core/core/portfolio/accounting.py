from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionAccounting:
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def buy(self, qty: int, exec_price: float, commission: float) -> None:
        if qty <= 0:
            return
        notional = float(qty) * float(exec_price)
        total_cost = notional + float(commission)
        new_qty = self.quantity + int(qty)
        if new_qty <= 0:
            return
        self.avg_cost = ((self.quantity * self.avg_cost) + total_cost) / float(new_qty)
        self.quantity = new_qty

    def sell(self, qty: int, exec_price: float, commission: float, sell_tax: float) -> float:
        if qty <= 0:
            return 0.0
        clamped_qty = min(int(qty), int(self.quantity))
        if clamped_qty <= 0:
            return 0.0

        gross = float(clamped_qty) * float(exec_price)
        pnl = (float(exec_price) - self.avg_cost) * float(clamped_qty) - commission - sell_tax
        self.quantity -= clamped_qty
        self.realized_pnl += pnl
        if self.quantity == 0:
            self.avg_cost = 0.0
        return gross

    def unrealized_pnl(self, mark_price: float) -> float:
        return (float(mark_price) - self.avg_cost) * float(self.quantity)

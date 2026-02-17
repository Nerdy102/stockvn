from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def compute_commission(notional: float, rate: float) -> float:
    return float(max(0.0, notional) * max(0.0, rate))


def compute_sell_tax(gross_sell_value: float, rate: float = 0.001) -> float:
    return float(max(0.0, gross_sell_value) * max(0.0, rate))


def compute_dividend_withholding(div_cash: float, rate: float = 0.05) -> float:
    return float(max(0.0, div_cash) * max(0.0, rate))


@dataclass(frozen=True)
class CostBreakdown:
    gross_pnl: float
    commission: float
    sell_tax: float
    dividend_withholding: float
    slippage_cost: float
    net_pnl: float


@dataclass(frozen=True)
class FeesTaxes:
    sell_tax_rate: float
    dividend_tax_rate: float
    default_commission_rate: float
    broker_commission: dict[str, float]

    @staticmethod
    def from_yaml(path: str | Path) -> FeesTaxes:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        taxes = data.get("taxes", {}) or {}
        fees = data.get("fees", {}) or {}
        brokers = fees.get("brokers", {}) or {}

        broker_commission: dict[str, float] = {
            str(k): float((v or {}).get("commission_rate", 0.0)) for k, v in brokers.items()
        }

        return FeesTaxes(
            sell_tax_rate=float(taxes.get("capital_gains_sell_tax_rate", 0.001)),
            dividend_tax_rate=float(taxes.get("dividend_tax_rate", 0.05)),
            default_commission_rate=float(fees.get("default_commission_rate", 0.0015)),
            broker_commission=broker_commission,
        )

    def commission_rate(self, broker_name: str | None = None) -> float:
        if broker_name and broker_name in self.broker_commission:
            return self.broker_commission[broker_name]
        return self.default_commission_rate

    def commission(self, notional: float, broker_name: str | None = None) -> float:
        return compute_commission(notional, self.commission_rate(broker_name))

    def sell_tax(self, sell_notional: float) -> float:
        return compute_sell_tax(sell_notional, self.sell_tax_rate)

    def dividend_tax(self, dividend_cash: float) -> float:
        return compute_dividend_withholding(dividend_cash, self.dividend_tax_rate)

    def build_pnl_breakdown(
        self,
        *,
        gross_pnl: float,
        commission: float,
        sell_tax: float,
        dividend_withholding: float = 0.0,
        slippage_cost: float = 0.0,
    ) -> CostBreakdown:
        net = gross_pnl - commission - sell_tax - dividend_withholding - slippage_cost
        return CostBreakdown(
            gross_pnl=float(gross_pnl),
            commission=float(commission),
            sell_tax=float(sell_tax),
            dividend_withholding=float(dividend_withholding),
            slippage_cost=float(slippage_cost),
            net_pnl=float(net),
        )

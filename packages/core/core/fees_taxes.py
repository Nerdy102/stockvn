from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class FeesTaxes:
    """Fees & taxes config."""

    sell_tax_rate: float
    dividend_tax_rate: float
    default_commission_rate: float
    broker_commission: Dict[str, float]

    @staticmethod
    def from_yaml(path: str | Path) -> "FeesTaxes":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        taxes = data.get("taxes", {}) or {}
        fees = data.get("fees", {}) or {}
        brokers = fees.get("brokers", {}) or {}

        broker_commission: Dict[str, float] = {
            str(k): float((v or {}).get("commission_rate", 0.0)) for k, v in brokers.items()
        }

        return FeesTaxes(
            sell_tax_rate=float(taxes.get("capital_gains_sell_tax_rate", 0.001)),
            dividend_tax_rate=float(taxes.get("dividend_tax_rate", 0.05)),
            default_commission_rate=float(fees.get("default_commission_rate", 0.0015)),
            broker_commission=broker_commission,
        )

    def commission_rate(self, broker_name: Optional[str] = None) -> float:
        if broker_name and broker_name in self.broker_commission:
            return self.broker_commission[broker_name]
        return self.default_commission_rate

    def commission(self, notional: float, broker_name: Optional[str] = None) -> float:
        return float(notional) * self.commission_rate(broker_name)

    def sell_tax(self, sell_notional: float) -> float:
        return float(sell_notional) * self.sell_tax_rate

    def dividend_tax(self, dividend_cash: float) -> float:
        return float(dividend_cash) * self.dividend_tax_rate

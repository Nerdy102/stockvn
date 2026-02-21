from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySpec:
    name: str
    group: str
    variant: dict[str, float | str | bool]


def build_strategy_registry(raocmoe_available: bool) -> list[StrategySpec]:
    out = [
        StrategySpec("Baseline_BH_EW", "G0", {}),
        StrategySpec("Baseline_EW_Monthly", "G0", {}),
        StrategySpec("Baseline_MOM20", "G0", {}),
        StrategySpec("Baseline_MA_Cross", "G0", {}),
        StrategySpec("Strategy_USER", "G1", {}),
    ]
    if raocmoe_available:
        for n in [
            "RAOCMOE_FULL",
            "RAOCMOE_minus_D1",
            "RAOCMOE_minus_D2",
            "RAOCMOE_minus_D3",
            "RAOCMOE_minus_D4",
            "RAOCMOE_minus_D5",
            "RAOCMOE_minus_D6",
        ]:
            out.append(StrategySpec(n, "G2", {}))
    return out

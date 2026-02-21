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
        StrategySpec("Strategy_USER", "G1", {"alias": "USER_V0"}),
        StrategySpec("USER_V0", "G1", {}),
        StrategySpec("USER_V1_STABILITY", "G1", {}),
        StrategySpec("USER_V2_COSTAWARE", "G1", {}),
        StrategySpec("USER_V3_REGIME_UQ", "G1", {}),
    ]
    if raocmoe_available:
        out.extend(
            [
                StrategySpec("RAOCMOE_FULL", "G2", {}),
                StrategySpec("RAOCMOE_minus_D1", "G2", {}),
                StrategySpec("RAOCMOE_minus_D2", "G2", {}),
                StrategySpec("RAOCMOE_minus_D3", "G2", {}),
                StrategySpec("RAOCMOE_minus_D4", "G2", {}),
                StrategySpec("RAOCMOE_minus_D5", "G2", {}),
                StrategySpec("RAOCMOE_minus_D6", "G2", {}),
            ]
        )
    return out

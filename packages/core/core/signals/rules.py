from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinRuleSpec:
    name: str
    signature: str


BUILTIN_RULES = [
    BuiltinRuleSpec("SMA", "SMA(n) | SMA(series,n)"),
    BuiltinRuleSpec("EMA", "EMA(n) | EMA(series,n)"),
    BuiltinRuleSpec("RSI", "RSI(n) | RSI(series,n)"),
    BuiltinRuleSpec("AVG", "AVG(series,n)"),
    BuiltinRuleSpec("CROSSOVER", "CROSSOVER(a,b)"),
    BuiltinRuleSpec("CROSSUNDER", "CROSSUNDER(a,b)"),
]

from __future__ import annotations


def disclaimer() -> str:
    return "past ≠ future, overfit risk, liquidity/limit risk"


def build_metrics_table(metrics: dict) -> list[dict]:
    keys = ["CAGR", "MDD", "Sharpe", "turnover", "costs"]
    return [{"metric": k, "value": float(metrics.get(k, 0.0))} for k in keys]

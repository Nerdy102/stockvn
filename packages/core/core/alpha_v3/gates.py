from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_GATES_PATH = Path("configs/research_gates.yaml")


def load_research_gates(path: Path = DEFAULT_GATES_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate_research_gates(
    dsr_value: float,
    pbo_phi: float,
    turnover: float,
    capacity_avg_order_notional_over_adtv: float,
    gates_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = gates_cfg or load_research_gates()
    rules = cfg.get("research_gates", cfg)

    reasons: list[str] = []
    dsr_min = float(rules.get("dsr_min", 0.95))
    pbo_max = float(rules.get("pbo_max", 0.10))
    turnover_max = float(rules.get("turnover_max", 1e9))
    capacity_max = float(rules.get("capacity_max", 1.0))

    if dsr_value < dsr_min:
        reasons.append(f"DSR {dsr_value:.4f} < {dsr_min:.4f}")
    if pbo_phi > pbo_max:
        reasons.append(f"PBO {pbo_phi:.4f} > {pbo_max:.4f}")
    if turnover > turnover_max:
        reasons.append(f"Turnover {turnover:.4f} > {turnover_max:.4f}")
    if capacity_avg_order_notional_over_adtv > capacity_max:
        reasons.append(
            "Capacity avg_order_notional_over_adtv "
            f"{capacity_avg_order_notional_over_adtv:.4f} > {capacity_max:.4f}"
        )

    status = "PASS" if not reasons else "FAIL"
    return {
        "status": status,
        "reasons": reasons,
        "thresholds": {
            "dsr_min": dsr_min,
            "pbo_max": pbo_max,
            "turnover_max": turnover_max,
            "capacity_max": capacity_max,
        },
    }

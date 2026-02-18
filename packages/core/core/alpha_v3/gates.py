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
    psr_value: float,
    rc_p_value: float,
    spa_p_value: float,
    gates_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = gates_cfg or load_research_gates()
    rules = cfg.get("research_gates", cfg)

    reasons: list[str] = []
    dsr_min = float(rules.get("dsr_min", 0.95))
    pbo_max = float(rules.get("pbo_max", 0.10))
    psr_min = float(rules.get("psr_min", 0.95))
    rc_p_max = float(rules.get("rc_p_max", 0.10))
    spa_p_max = float(rules.get("spa_p_max", 0.10))

    if dsr_value < dsr_min:
        reasons.append(f"DSR {dsr_value:.4f} < {dsr_min:.4f}")
    if pbo_phi > pbo_max:
        reasons.append(f"PBO {pbo_phi:.4f} > {pbo_max:.4f}")
    if psr_value < psr_min:
        reasons.append(f"PSR {psr_value:.4f} < {psr_min:.4f}")
    if rc_p_value > rc_p_max:
        reasons.append(f"RC_p {rc_p_value:.4f} > {rc_p_max:.4f}")
    if spa_p_value > spa_p_max:
        reasons.append(f"SPA_p {spa_p_value:.4f} > {spa_p_max:.4f}")

    status = "PASS" if not reasons else "FAIL"
    return {
        "status": status,
        "reasons": reasons,
        "thresholds": {
            "dsr_min": dsr_min,
            "pbo_max": pbo_max,
            "psr_min": psr_min,
            "rc_p_max": rc_p_max,
            "spa_p_max": spa_p_max,
        },
    }

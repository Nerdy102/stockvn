from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlmodel import Session

from core.db.models import DataHealthIncident

RUNBOOK_SECTION_GOVERNANCE = "GOV-OPS-001"


@dataclass(frozen=True)
class GovernanceV3Policy:
    # locked promotion rules
    min_sharpe_delta: float = 0.05
    max_mdd_increase: float = 0.02
    require_drift_ok: bool = True
    require_capacity_ok: bool = True

    # locked pause triggers
    max_ece: float = 0.08
    max_psi: float = 0.25
    max_cash_recon_gap: float = 1.0

    # locked forced-cash fail-safe
    forced_cash_weight_on_pause: float = 1.0


def evaluate_promotion_v3(
    champion: dict[str, float],
    challenger: dict[str, float],
    *,
    drift_ok: bool,
    capacity_ok: bool,
    policy: GovernanceV3Policy = GovernanceV3Policy(),
) -> dict[str, Any]:
    c_sh = float(champion.get("sharpe_net", 0.0))
    n_sh = float(challenger.get("sharpe_net", 0.0))
    c_mdd = float(champion.get("max_drawdown", 0.0))
    n_mdd = float(challenger.get("max_drawdown", 0.0))

    checks = {
        "sharpe_delta_ok": (n_sh - c_sh) >= policy.min_sharpe_delta,
        "mdd_not_worse_ok": (n_mdd - c_mdd) <= policy.max_mdd_increase,
        "drift_ok": (drift_ok if policy.require_drift_ok else True),
        "capacity_ok": (capacity_ok if policy.require_capacity_ok else True),
    }
    promote = bool(all(checks.values()))
    return {
        "promote": promote,
        "decision": "PROMOTE_CHALLENGER" if promote else "KEEP_CHAMPION",
        "checks": checks,
        "champion": {"sharpe_net": c_sh, "max_drawdown": c_mdd},
        "challenger": {"sharpe_net": n_sh, "max_drawdown": n_mdd},
    }


def evaluate_pause_triggers_v3(
    *,
    ece: float,
    psi_max: float,
    cash_recon_gap_abs: float,
    severe_failure: bool,
    policy: GovernanceV3Policy = GovernanceV3Policy(),
) -> dict[str, Any]:
    reasons: list[str] = []
    if float(ece) > policy.max_ece:
        reasons.append("ece_above_threshold")
    if float(psi_max) > policy.max_psi:
        reasons.append("psi_above_threshold")
    if float(cash_recon_gap_abs) > policy.max_cash_recon_gap:
        reasons.append("cash_reconciliation_break")
    if bool(severe_failure):
        reasons.append("severe_failure")

    paused = bool(len(reasons) > 0)
    return {
        "paused": paused,
        "rollback_to_champion": paused,
        "reasons": reasons,
        "forced_cash_weight": float(policy.forced_cash_weight_on_pause) if paused else 0.0,
    }


def apply_governance_fail_safe_v3(
    target_weights: np.ndarray,
    order_intents: list[dict[str, Any]],
    pause_state: dict[str, Any],
    *,
    policy: GovernanceV3Policy = GovernanceV3Policy(),
) -> tuple[np.ndarray, float, list[dict[str, Any]], dict[str, Any]]:
    w = np.asarray(target_weights, dtype=float)
    if not bool(pause_state.get("paused", False)):
        cash = max(0.0, 1.0 - float(np.sum(w)))
        return w, cash, order_intents, {"mode": "normal", "paused": False}

    blocked = []
    for it in order_intents:
        row = dict(it)
        row["blocked_by_governance"] = True
        row["blocked_reason"] = ",".join([str(x) for x in pause_state.get("reasons", [])])
        blocked.append(row)

    safe = np.zeros_like(w)
    return safe, float(policy.forced_cash_weight_on_pause), [], {
        "mode": "forced_cash",
        "paused": True,
        "rollback_to_champion": bool(pause_state.get("rollback_to_champion", True)),
        "blocked_orders": blocked,
        "reasons": list(pause_state.get("reasons", [])),
    }


def create_governance_incident_v3(
    session: Session,
    *,
    severity: str,
    summary: str,
    details: dict[str, Any],
    symbol: str | None = None,
) -> DataHealthIncident:
    sev = str(severity).upper()
    if sev not in {"LOW", "MEDIUM", "HIGH", "SEVERE"}:
        sev = "HIGH"

    row = DataHealthIncident(
        source="governance_v3",
        severity=sev,
        status="OPEN",
        symbol=symbol,
        summary=summary,
        details_json=dict(details),
        runbook_section=RUNBOOK_SECTION_GOVERNANCE,
        suggested_actions_json={
            "actions": [
                "Pause order generation immediately",
                "Rollback to champion model",
                "Investigate drift/calibration and reconcile trade/cash logs",
            ]
        },
        created_at=dt.datetime.utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row

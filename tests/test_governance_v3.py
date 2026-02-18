from __future__ import annotations

import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DataHealthIncident
from core.governance_v3 import (
    RUNBOOK_SECTION_GOVERNANCE,
    apply_governance_fail_safe_v3,
    create_governance_incident_v3,
    evaluate_pause_triggers_v3,
    evaluate_promotion_v3,
)


def test_governance_v3_pause_prevents_order_generation_and_forces_cash() -> None:
    target = np.array([0.25, 0.35, 0.20], dtype=float)
    intents = [
        {"symbol": "AAA", "side": "BUY", "qty": 100},
        {"symbol": "BBB", "side": "SELL", "qty": 100},
    ]

    pause = evaluate_pause_triggers_v3(
        ece=0.10,
        psi_max=0.10,
        cash_recon_gap_abs=0.0,
        severe_failure=False,
    )
    out_w, out_cash, out_intents, audit = apply_governance_fail_safe_v3(target, intents, pause)

    assert pause["paused"] is True
    assert np.allclose(out_w, np.zeros_like(target))
    assert out_cash == 1.0
    assert out_intents == []
    assert audit["mode"] == "forced_cash"
    assert audit["paused"] is True
    assert len(audit["blocked_orders"]) == 2


def test_governance_v3_incident_created_for_severe_failure() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        row = create_governance_incident_v3(
            session,
            severity="SEVERE",
            summary="Severe governance trigger: reconciliation breach",
            details={"cash_recon_gap_abs": 12.5, "pause": True, "rollback": True},
            symbol="VN30",
        )

        got = session.exec(select(DataHealthIncident).where(DataHealthIncident.id == row.id)).first()
        assert got is not None
        assert got.source == "governance_v3"
        assert got.severity == "SEVERE"
        assert got.status == "OPEN"
        assert got.runbook_section == RUNBOOK_SECTION_GOVERNANCE
        assert got.details_json["rollback"] is True


def test_governance_v3_promotion_rules_locked() -> None:
    champion = {"sharpe_net": 1.10, "max_drawdown": -0.12}
    challenger = {"sharpe_net": 1.20, "max_drawdown": -0.11}

    ok = evaluate_promotion_v3(champion, challenger, drift_ok=True, capacity_ok=True)
    assert ok["promote"] is True

    bad = evaluate_promotion_v3(champion, challenger, drift_ok=False, capacity_ok=True)
    assert bad["promote"] is False
    assert bad["decision"] == "KEEP_CHAMPION"

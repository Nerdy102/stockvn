from __future__ import annotations

from core.alpha_v3.gates import evaluate_research_gates


def test_research_gate_pass_fail_reasons() -> None:
    passed = evaluate_research_gates(0.97, 0.05, turnover=1.0, capacity_avg_order_notional_over_adtv=0.1)
    assert passed["status"] == "PASS"
    assert passed["reasons"] == []

    failed = evaluate_research_gates(0.60, 0.40, turnover=3.0, capacity_avg_order_notional_over_adtv=0.5)
    assert failed["status"] == "FAIL"
    assert len(failed["reasons"]) >= 3

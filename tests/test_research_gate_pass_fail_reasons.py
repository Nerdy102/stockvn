from __future__ import annotations

from core.alpha_v3.gates import evaluate_research_gates


def test_research_gate_pass_fail_reasons() -> None:
    passed = evaluate_research_gates(0.97, 0.05, psr_value=0.97, rc_p_value=0.05, spa_p_value=0.05)
    assert passed["status"] == "PASS"
    assert passed["reasons"] == []

    failed = evaluate_research_gates(0.60, 0.40, psr_value=0.70, rc_p_value=0.30, spa_p_value=0.20)
    assert failed["status"] == "FAIL"
    assert len(failed["reasons"]) == 5

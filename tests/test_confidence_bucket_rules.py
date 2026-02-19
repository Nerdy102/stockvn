from __future__ import annotations

from core.simple_mode.confidence import confidence_bucket


def test_confidence_bucket_rules_stable() -> None:
    c1, s1 = confidence_bucket(
        has_min_rows=True,
        liquidity="vừa",
        regime="risk_on",
        regime_expected="risk_on",
        atr_pct=0.02,
    )
    c2, s2 = confidence_bucket(
        has_min_rows=True,
        liquidity="vừa",
        regime="risk_on",
        regime_expected="risk_on",
        atr_pct=0.02,
    )
    assert (c1, s1) == (c2, s2)
    assert c1 == "Cao"

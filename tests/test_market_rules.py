from __future__ import annotations

from core.market_rules import load_market_rules


def test_tick_rounding() -> None:
    rules = load_market_rules("configs/market_rules_vn.yaml")
    assert rules.round_price(9990) % 10 == 0
    assert rules.round_price(10000) % 50 == 0
    assert rules.round_price(50000) % 100 == 0
    assert rules.get_tick_size(49950) == 50
    assert rules.get_tick_size(50000) == 100


def test_price_limit_validation() -> None:
    rules = load_market_rules("configs/market_rules_vn.yaml")
    ref = 10000.0
    assert rules.validate_price_limit(10700.0, ref, "normal")
    assert not rules.validate_price_limit(10800.0, ref, "normal")

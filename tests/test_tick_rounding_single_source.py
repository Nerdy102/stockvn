from core.market_rules import load_market_rules


def test_tick_rounding_single_source() -> None:
    rules = load_market_rules('configs/market_rules_vn.yaml')
    assert rules.round_price(10012, direction='up') == 10050
    assert rules.round_price(10049, direction='down') == 10000

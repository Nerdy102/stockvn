from core.market_rules import load_market_rules


def test_tick_rounding_bands_and_instruments() -> None:
    rules = load_market_rules("configs/market_rules_vn.yaml")
    assert rules.get_tick_size(9990) == 10
    assert rules.get_tick_size(10000) == 50
    assert rules.get_tick_size(50000) == 100
    assert rules.get_tick_size(20000, instrument="etf") == 10
    assert rules.get_tick_size(20000, put_through=True) == 1

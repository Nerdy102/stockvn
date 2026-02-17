from core.market_rules import load_market_rules


def test_price_limits_normal_and_special() -> None:
    rules = load_market_rules("configs/market_rules_vn.yaml")
    lo, hi = rules.calc_price_limits(10000, context="normal")
    assert lo == 9300
    assert hi == 10700
    assert rules.price_limit_pct("first_trading_day") == 0.2

from core.market_rules import get_reference_price


def test_get_reference_price_adjusted_by_split_and_rights() -> None:
    ref = get_reference_price(
        12000.0,
        [
            {"action_type": "SPLIT", "params_json": {"split_factor": 2.0}},
            {"action_type": "RIGHTS_ISSUE", "params_json": {"ratio": 0.2, "subscription_price": 4000.0}},
        ],
    )
    assert ref == ((12000.0 / 2.0) + 0.2 * 4000.0) / 1.2

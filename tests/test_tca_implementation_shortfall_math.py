from core.tca.service import _quality


def test_buy_sell_sign_and_bucket_math() -> None:
    buy_is = (11 - 10) * 100
    sell_is = -1 * (9 - 10) * 100
    assert buy_is > 0
    assert sell_is > 0
    is_total = 120.0
    notional = 100_000.0
    bps = (is_total / notional) * 10000
    assert round(bps, 2) == 12.0
    q, _ = _quality(bps)
    assert q == "Vừa"

def test_twap_mean_price_ref() -> None:
    refs = [10.0, 11.0, 12.0]
    twap = sum(refs) / len(refs)
    assert twap == 11.0


def test_vwap_typical_price_weighted() -> None:
    highs = [11.0, 12.0]
    lows = [9.0, 10.0]
    closes = [10.0, 11.0]
    vols = [100.0, 300.0]
    typical = [ (h+l+c)/3.0 for h,l,c in zip(highs,lows,closes)]
    vwap = sum(t*v for t,v in zip(typical, vols)) / sum(vols)
    assert round(vwap, 4) == round((10.0*100 + 11.0*300)/400, 4)

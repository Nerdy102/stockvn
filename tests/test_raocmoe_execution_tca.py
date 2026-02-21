from core.raocmoe.execution_tca import ExecutionTCA


def test_execution_fill_penalty() -> None:
    cfg = {
        "execution": {
            "tca": {"lr": 0.05, "huber_delta": 1.5},
            "tca_cpd": {"threshold_base": 8.5, "cooldown_bars": 12},
            "limit_fill_penalty": {"buy_at_upper_limit_mult": 0.2, "sell_at_lower_limit_mult": 0.2},
        }
    }
    tca = ExecutionTCA(cfg)
    out = tca.estimate(
        symbol="AAA",
        side="BUY",
        delta_weight=0.2,
        nav=1_000_000,
        price=10123,
        adtv=100_000_000,
        atr14=20.0,
        instrument_type="stock",
        reference_price=10_000,
        at_upper_limit=True,
    )
    assert out.fill_ratio <= 0.2 + 1e-9
    assert float(out.qty).is_integer()

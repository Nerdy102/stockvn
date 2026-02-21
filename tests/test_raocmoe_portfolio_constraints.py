from core.raocmoe.portfolio_controller import PortfolioController


def test_portfolio_constraints() -> None:
    cfg = {
        "moe": {"outputs": {"mu_clip": 0.1}},
        "portfolio": {
            "target_cash_min": 0.1,
            "max_single_weight": 0.1,
            "max_sector_weight": 0.25,
            "turnover_cap_l1": 0.25,
            "no_trade_band_abs": 0.002,
            "liquidity": {"participation_limit": 0.05, "days_to_exit": 3},
            "risk": {"vol_target_annual": 0.18},
            "robust": {"uncertainty_penalty_lambda": 0.35, "worst_case": True},
        },
    }
    p = PortfolioController(cfg)
    universe = ["A", "B", "C", "D"]
    out = p.build_target(
        universe,
        {s: 0.05 for s in universe},
        {s: 0.01 for s in universe},
        {s: 0.0 for s in universe},
        {s: 0.02 for s in universe},
        {s: 1e7 for s in universe},
        nav=1e6,
        regime="SIDEWAYS",
        sectors={"A": "X", "B": "X", "C": "Y", "D": "Y"},
    )
    assert out.cash_weight >= 0.1
    assert max(out.weights.values()) <= 0.1 + 1e-9
    assert sum(abs(out.weights[s]) for s in universe) <= 0.9 + 1e-6

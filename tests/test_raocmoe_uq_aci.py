import numpy as np

from core.raocmoe.uq import UncertaintyEngine


def test_uq_aci_converges() -> None:
    cfg = {
        "alpha_target_horizons": [{"name": "h1", "steps": 1, "target_miscoverage": 0.10}],
        "uq": {
            "aci": {
                "gamma_base": 0.02,
                "gamma_min": 0.005,
                "gamma_max": 0.08,
                "adaptive_gamma": {"enabled": True, "psi_weight": 0.25, "cpd_weight": 0.25},
            },
            "pooling": {"min_pool_size": 5},
            "regime_calibrators": {"max_scores_per_pool": 200, "warm_start_scores": 10},
            "coverage_monitor": {
                "rolling_window": 100,
                "undercoverage_tolerance": 0.03,
                "consecutive_breaches_to_pause": 3,
            },
        },
    }
    uq = UncertaintyEngine(cfg)
    rng = np.random.default_rng(7)
    misses = []
    for t in range(1, 600):
        y = float(rng.normal(0.0, 1.0))
        ivs, _ = uq.get_intervals(t, "SIDEWAYS", "BTC", "OTHER", 0.0, 1.0)
        iv = ivs[0]
        misses.append(int(not (iv.lower <= y <= iv.upper)))
        uq.update_with_label(t, "h1", y, psi=0.0, cp_score=0.0, sector="OTHER")
    miss = float(np.mean(misses[-250:]))
    assert abs(miss - 0.10) < 0.10

import pandas as pd

from scripts.run_improvement_suite import choose_dev_winner, objective_score


def test_objective_score_formula() -> None:
    row = pd.Series(
        {
            "sharpe": 1.0,
            "stress_fragility_score": 0.4,
            "turnover_l1_annualized": 10.0,
            "mdd": -0.2,
        }
    )
    out = objective_score(row)
    assert abs(out - (1.0 - 0.2 - 0.5 - 0.04)) < 1e-12


def test_choose_dev_winner_only_pass_variants() -> None:
    df = pd.DataFrame(
        [
            {"strategy": "USER_V0", "variant_pass": 1, "objective_score": 0.2},
            {"strategy": "USER_V1_STABILITY", "variant_pass": 1, "objective_score": 0.4},
            {"strategy": "USER_V2_COSTAWARE", "variant_pass": 0, "objective_score": 999.0},
            {"strategy": "Baseline_BH_EW", "variant_pass": 1, "objective_score": 1000.0},
        ]
    )
    assert choose_dev_winner(df) == "USER_V1_STABILITY"

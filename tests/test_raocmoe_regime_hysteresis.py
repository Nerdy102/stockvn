from core.raocmoe.regime import RegimeEngine


def test_regime_hysteresis_no_flip_flop() -> None:
    cfg = {
        "regime": {
            "states": ["TREND_UP", "SIDEWAYS", "RISK_OFF", "PANIC_VOL"],
            "hysteresis": {"p_on": 0.65, "p_off": 0.45, "cooldown_bars": 8},
            "transition_matrix": {
                "TREND_UP": {
                    "TREND_UP": 0.85,
                    "SIDEWAYS": 0.1,
                    "RISK_OFF": 0.04,
                    "PANIC_VOL": 0.01,
                },
                "SIDEWAYS": {
                    "TREND_UP": 0.12,
                    "SIDEWAYS": 0.75,
                    "RISK_OFF": 0.1,
                    "PANIC_VOL": 0.03,
                },
                "RISK_OFF": {"TREND_UP": 0.05, "SIDEWAYS": 0.15, "RISK_OFF": 0.7, "PANIC_VOL": 0.1},
                "PANIC_VOL": {
                    "TREND_UP": 0.03,
                    "SIDEWAYS": 0.07,
                    "RISK_OFF": 0.2,
                    "PANIC_VOL": 0.7,
                },
            },
            "feature_weights": {
                k: {
                    "trend_strength": 0.1,
                    "realized_vol": 0.0,
                    "corr_index": 0.0,
                    "liq_stress": 0.0,
                    "cpd_score": 0.0,
                }
                for k in ["TREND_UP", "SIDEWAYS", "RISK_OFF", "PANIC_VOL"]
            },
        }
    }
    eng = RegimeEngine(cfg)
    hard = []
    for t in range(1, 40):
        p = eng.update(
            t,
            {
                "trend_strength": 0.01 if t % 2 else -0.01,
                "realized_vol": 0.1,
                "corr_index": 0.1,
                "liq_stress": 0.1,
                "cpd_score": 0.0,
            },
        )
        hard.append(p.hard_regime)
    changes = sum(1 for i in range(1, len(hard)) if hard[i] != hard[i - 1])
    assert changes < 5

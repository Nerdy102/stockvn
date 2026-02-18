from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.ml.meta_allocator_v3 import meta_allocate_v3


def test_meta_allocator_v3_penalizes_cvar_and_cost() -> None:
    experts = pd.DataFrame(
        {
            "expert": ["cheap_low_tail", "expensive_high_tail"],
            "r21": [0.02, 0.022],
            "cvar": [0.01, 0.03],
            "cost_bps": [5.0, 40.0],
        }
    )

    weights, audit = meta_allocate_v3(experts)

    assert float(weights["cheap_low_tail"]) > float(weights["expensive_high_tail"])
    assert audit["formula"] == "u = r21 - 3*cvar - 0.5*(cost_bps/10000)"
    assert len(audit["components"]) == 2
    assert {row["expert"] for row in audit["components"]} == {"cheap_low_tail", "expensive_high_tail"}


def test_meta_allocator_v3_two_expert_toy_matches_golden_direction() -> None:
    toy = json.loads(Path("tests/golden/meta_allocator_v3_toy_2experts.json").read_text(encoding="utf-8"))
    experts = pd.DataFrame(toy["experts"])

    weights, audit = meta_allocate_v3(experts)

    winner = toy["expected_higher_weight_expert"]
    loser = [x for x in experts["expert"].tolist() if x != winner][0]
    assert float(weights[winner]) > float(weights[loser])
    assert all("utility" in row and "weight" in row and "eg_weight_raw" in row for row in audit["components"])


def test_meta_allocator_v3_deterministic() -> None:
    experts = pd.DataFrame(
        {
            "expert": ["a", "b", "c"],
            "r21": [0.01, 0.012, 0.009],
            "cvar": [0.015, 0.02, 0.01],
            "cost_bps": [8.0, 12.0, 6.0],
        }
    )

    w1, a1 = meta_allocate_v3(experts)
    w2, a2 = meta_allocate_v3(experts)

    pd.testing.assert_series_equal(w1.sort_index(), w2.sort_index())
    assert a1 == a2

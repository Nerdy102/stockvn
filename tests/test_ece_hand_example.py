from __future__ import annotations

import json
from pathlib import Path

from core.alpha_v3.calibration import compute_probability_calibration_metrics


def test_ece_hand_example() -> None:
    golden = json.loads(Path("tests/golden/ece_hand_example.json").read_text())
    probs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    outcomes = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]

    out = compute_probability_calibration_metrics(probs, outcomes, bins=10)
    assert round(float(out["brier"]), 3) == golden["expected"]["brier"]
    assert round(float(out["ece"]), 2) == golden["expected"]["ece"]
    assert len(out["reliability_bins_json"]) == golden["expected"]["bins"]

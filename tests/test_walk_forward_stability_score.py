from __future__ import annotations

from core.validation.walk_forward import summarize_walk_forward


def test_walk_forward_stability_score() -> None:
    out = summarize_walk_forward([0.01, 0.015, 0.02, 0.013])
    assert out["stability"] in {"Cao", "Vừa", "Thấp"}

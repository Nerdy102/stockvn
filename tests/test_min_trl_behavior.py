from __future__ import annotations

from core.quant_stats.psr_dsr import min_track_record_length


def test_min_trl_behavior() -> None:
    near, _ = min_track_record_length(0.11, 0.10, 0.0, 3.0)
    far, _ = min_track_record_length(0.40, 0.10, 0.0, 3.0)
    assert near is not None and far is not None
    assert near > far

    none_case, reason = min_track_record_length(0.05, 0.10, 0.0, 3.0)
    assert none_case is None
    assert "không vượt" in reason.lower()

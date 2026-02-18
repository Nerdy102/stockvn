from __future__ import annotations

from core.monitoring.data_health import compute_schema_diff


def test_schema_diff_correct_on_toy() -> None:
    old_keys = {"symbol", "price", "ts", "volume"}
    new_keys = {"symbol", "price", "timestamp", "value_vnd"}
    out = compute_schema_diff(old_keys, new_keys)
    assert out["added_keys"] == ["timestamp", "value_vnd"]
    assert out["removed_keys"] == ["ts", "volume"]

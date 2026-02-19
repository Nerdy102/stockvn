from __future__ import annotations

from core.risk.sizing import compute_position_size


def test_position_sizing_board_lot() -> None:
    qty, reason = compute_position_size(nav=1_000_000, close=10000, atr14=120, market="vn")
    assert reason is None
    assert int(qty) % 100 == 0

from __future__ import annotations

from core.risk.sizing import round_down_to_board_lot


def test_tick_lot_rounding_vn() -> None:
    assert round_down_to_board_lot(255, lot=100) == 200

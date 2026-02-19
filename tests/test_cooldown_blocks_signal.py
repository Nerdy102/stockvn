from __future__ import annotations


def apply_cooldown(last_trade_bar: int, now_bar: int, cooldown_bars: int) -> str:
    return "TRUNG_TINH" if (now_bar - last_trade_bar) < cooldown_bars else "KEEP"


def test_cooldown_blocks_signal() -> None:
    assert apply_cooldown(100, 105, 10) == "TRUNG_TINH"

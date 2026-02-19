from __future__ import annotations

from core.simple_mode.briefing import build_market_brief


def test_briefing_output_vi_not_empty() -> None:
    out = build_market_brief(
        as_of_date="2026-02-19",
        breadth={"up": 12, "down": 10, "flat": 8},
        benchmark_change=0.56,
        volume_ratio=1.1,
        volatility_proxy=1.3,
    )
    assert out.strip()
    assert "thị trường" in out
    assert "độ" in out

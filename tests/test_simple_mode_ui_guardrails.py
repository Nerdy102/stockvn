from __future__ import annotations

from pathlib import Path


def test_simple_mode_font_stack_vietnamese_fallback() -> None:
    content = Path("apps/dashboard_streamlit/pages/simple_mode.py").read_text(encoding="utf-8")
    assert 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif' in content


def test_simple_mode_has_vietnamese_diacritic_smoke_text() -> None:
    content = Path("apps/dashboard_streamlit/pages/simple_mode.py").read_text(encoding="utf-8")
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư" in content

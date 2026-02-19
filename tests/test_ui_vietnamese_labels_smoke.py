from __future__ import annotations

from pathlib import Path


def test_ui_vietnamese_labels_smoke() -> None:
    content = Path("apps/dashboard_streamlit/pages/0_Home_Dashboard.py").read_text(encoding="utf-8")
    assert "Tình hình thị trường hôm nay (Market today)" in content
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư." in content

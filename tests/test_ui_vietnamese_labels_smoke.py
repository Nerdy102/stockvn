from __future__ import annotations

from pathlib import Path


def test_ui_vietnamese_labels_smoke() -> None:
    content = Path("apps/dashboard_streamlit/pages/0_Tong_quan_hom_nay.py").read_text(encoding="utf-8")
    assert "🏠 Tổng quan hôm nay" in content
    assert "Tình hình thị trường hôm nay (Market today)" in content
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư." in content

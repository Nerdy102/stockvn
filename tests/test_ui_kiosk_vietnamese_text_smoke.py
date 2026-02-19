from __future__ import annotations

from pathlib import Path


def test_ui_kiosk_vietnamese_text_smoke() -> None:
    content = Path("apps/web_kiosk/app.py").read_text(encoding="utf-8")
    assert "🏠 Hôm nay" in content
    assert "Xem tín hiệu hôm nay" in content
    assert "Tạo lệnh nháp" in content
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư." in content

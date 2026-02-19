from __future__ import annotations

from pathlib import Path


def test_ui_kiosk_vietnamese_text_smoke() -> None:
    content = Path("apps/web_kiosk/app.py").read_text(encoding="utf-8")
    assert "Kiosk UI v3: siêu tối giản" in content
    assert "Một màn hình duy nhất" in content
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư." in content
    assert "Mở giao diện nâng cao (Advanced UI)" in content

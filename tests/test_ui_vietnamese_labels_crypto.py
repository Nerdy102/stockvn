from __future__ import annotations

from pathlib import Path


def test_ui_vietnamese_labels_crypto() -> None:
    content = Path("apps/dashboard_streamlit/pages/simple_mode.py").read_text(encoding="utf-8")
    assert "Thị trường (Market)" in content
    assert "Tiền mã hoá (Crypto)" in content
    assert "Giao ngay — giao dịch giấy (Spot paper)" in content
    assert "Hợp đồng vĩnh cửu — giao dịch giấy (Perp paper, Long/Short)" in content

from __future__ import annotations

from pathlib import Path


def test_ui_live_labels_and_font_stack_present() -> None:
    home = Path("apps/dashboard_streamlit/pages/0_Tong_quan_hom_nay.py").read_text(encoding="utf-8")
    simple = Path("apps/dashboard_streamlit/pages/simple_mode.py").read_text(encoding="utf-8")
    assert 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif' in home
    assert "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư." in home
    assert "Trạng thái giao dịch thật (Live status)" in home
    assert "Kill-switch" in home
    assert "Xác nhận giao dịch thật (Live confirmation)" in simple
    assert "Kiểm tra trước khi gửi live" in simple


def test_ui_audit_log_page_vietnamese_labels() -> None:
    content = Path("apps/dashboard_streamlit/pages/12_Nhat_ky_he_thong.py").read_text(encoding="utf-8")
    assert "Nhật ký hệ thống (Audit log)" in content
    assert "Sức khoẻ hệ thống (System health)" in content

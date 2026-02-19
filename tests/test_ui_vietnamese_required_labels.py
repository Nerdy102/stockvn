from __future__ import annotations

from pathlib import Path


def test_ui_vietnamese_required_labels() -> None:
    content = Path('apps/web_kiosk/app.py').read_text(encoding='utf-8')
    content += Path('apps/web_kiosk/components/cards.py').read_text(encoding='utf-8')
    content += Path('apps/web_kiosk/components/actions.py').read_text(encoding='utf-8')

    required = [
        'Hôm nay (Today)',
        'Tín hiệu rõ ràng (Clear signals)',
        'Độ sẵn sàng (Readiness)',
        'Tạo lệnh nháp (Create draft)',
        'Xác nhận thực hiện (Confirm execute)',
        'Tôi hiểu có thể thua lỗ (Risk of loss)',
        'Tôi hiểu đây không phải lời khuyên đầu tư (Not investment advice)',
    ]
    for label in required:
        assert label in content

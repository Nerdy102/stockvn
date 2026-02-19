from __future__ import annotations

from typing import Any


def fallback_payload() -> dict[str, Any]:
    return {
        "as_of_date": "2026-02-19",
        "market_today_text": [
            "Thị trường đi ngang, biên độ vừa phải và thanh khoản ổn định.",
            "Nhóm vốn hóa lớn giữ nhịp, chưa có tín hiệu đảo chiều mạnh.",
            "Dòng tiền ưu tiên cổ phiếu có nền giá chặt và xu hướng tăng đều.",
            "Ưu tiên quản trị rủi ro: chia nhỏ lệnh, không dùng đòn bẩy cao.",
        ],
        "buy_candidates": [
            {
                "symbol": "FPT",
                "signal": "TĂNG",
                "confidence": "Cao",
                "reason": "Giá duy trì trên đường trung bình và khối lượng cải thiện.",
                "model_id": "model_1",
            },
            {
                "symbol": "MWG",
                "signal": "TĂNG",
                "confidence": "Vừa",
                "reason": "Đà hồi phục tốt, tín hiệu động lượng ngắn hạn tích cực.",
                "model_id": "model_2",
            },
        ],
        "sell_candidates": [
            {
                "symbol": "HPG",
                "signal": "GIẢM",
                "confidence": "Vừa",
                "reason": "Giá mất hỗ trợ ngắn hạn, lực bán chiếm ưu thế.",
                "model_id": "model_3",
            }
        ],
        "model_cards": [
            {
                "name": "Mô hình 1",
                "net_return_after_fees_taxes": 12.4,
                "max_drawdown": -8.1,
            },
            {
                "name": "Mô hình 2",
                "net_return_after_fees_taxes": 9.2,
                "max_drawdown": -6.4,
            },
            {
                "name": "Mô hình 3",
                "net_return_after_fees_taxes": 7.8,
                "max_drawdown": -5.9,
            },
        ],
        "paper_summary": {"pnl": 1520000, "trades_count": 14, "cash_pct": 62.5},
        "disclaimers": [
            "Đây là tín hiệu nghiên cứu (Research signal), không phải lời khuyên đầu tư (Not investment advice).",
            "Quá khứ không đảm bảo tương lai.",
            "Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư.",
        ],
    }

from __future__ import annotations


def build_market_brief(
    as_of_date: str,
    breadth: dict[str, int] | None,
    benchmark_change: float | None,
    volume_ratio: float | None,
    volatility_proxy: float | None,
) -> str:
    missing: list[str] = []
    if not breadth:
        missing.append("độ rộng thị trường")
    if benchmark_change is None:
        missing.append("biến động chỉ số")
    if volume_ratio is None:
        missing.append("thanh khoản")
    if volatility_proxy is None:
        missing.append("độ biến động")

    if missing:
        return (
            f"Bản tin ngày {as_of_date}: dữ liệu còn thiếu ({', '.join(missing)}).\n"
            "Hệ thống đang dùng chế độ an toàn, ưu tiên quan sát và chờ cập nhật thêm."
        )

    up = int((breadth or {}).get("up", 0))
    down = int((breadth or {}).get("down", 0))
    flat = int((breadth or {}).get("flat", 0))
    tone = "tăng" if benchmark_change > 0 else ("giảm" if benchmark_change < 0 else "đi ngang")
    liq_text = (
        "thấp hơn bình thường"
        if volume_ratio < 0.9
        else "cao hơn bình thường" if volume_ratio > 1.1 else "gần mức bình thường"
    )
    vol_text = (
        "dao động mạnh"
        if volatility_proxy > 2.0
        else "dao động vừa" if volatility_proxy > 1.0 else "dao động nhẹ"
    )

    return "\n".join(
        [
            f"Bản tin ngày {as_of_date}: thị trường đang {tone} ({benchmark_change:.2f}%).",
            f"Độ rộng: {up} mã tăng, {down} mã giảm, {flat} mã đi ngang.",
            f"Thanh khoản hiện {liq_text} (xấp xỉ {volume_ratio:.2f} lần trung bình 20 phiên).",
            f"Mức biến động hiện tại: {vol_text}.",
            "Khuyến nghị dễ hiểu: ưu tiên lệnh nháp nhỏ, tránh tất tay vì rủi ro luôn tồn tại.",
        ]
    )

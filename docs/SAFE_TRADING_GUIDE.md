# SAFE TRADING GUIDE — 1 trang thao tác an toàn

## Bạn bấm gì?
1. Chọn mã và mô hình.
2. Xem 3 dòng tóm tắt “Bạn sắp làm gì”.
3. Tích các ô xác nhận bắt buộc.
4. Bấm **Xác nhận thực hiện (Confirm execute)**.

## Hệ thống tự chặn gì?
- **Kill-switch 3 lớp**:
  - Cấu hình ENV (`KILL_SWITCH=true`) chặn live.
  - Cờ DB qua nút **Dừng khẩn cấp**.
  - Tự động dừng khi chạm ngưỡng rủi ro ngày.
- **Risk limits mặc định an toàn**:
  - `max_notional_per_order`
  - `max_orders_per_day`
  - `max_daily_loss`
- **Ngoài giờ giao dịch VN**: chỉ cho phép lưu **lệnh nháp (Draft)**.
- **Idempotency**: bấm 2 lần không tạo 2 lệnh giống nhau.

## Khi nào phải dừng?
- Khi thấy mã lỗi `RISK_BLOCKED`, `OFF_SESSION_DRAFT_ONLY`, `LIVE_BLOCKED_*`.
- Khi hệ thống đã bật trạng thái dừng khẩn cấp.
- Khi bạn không chắc về rủi ro và không hiểu lệnh chuẩn bị gửi.

## Nếu dưới 18 tuổi
- Chế độ **Live** bị khoá.
- Chỉ được dùng **Draft/Paper**.
- Cần người giám hộ/đủ điều kiện pháp lý khi muốn giao dịch thật.
- Không hỗ trợ hướng dẫn lách luật.

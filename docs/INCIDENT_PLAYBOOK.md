# Playbook sự cố giao dịch

## Khi lỗi gửi lệnh
1. Bật **DỪNG KHẨN CẤP (Kill-switch)** ngay lập tức.
2. Đóng quyền gửi lệnh live (đặt `ENABLE_LIVE_TRADING=false`).
3. Kiểm tra audit log `artifacts/audit/order_audit.jsonl` để xác định trạng thái cuối.
4. Đối soát với broker qua mã lệnh ngoài hệ thống (nếu có).

## Khi lỗi khớp lệnh hoặc timeout ACK
1. Đánh dấu lệnh trạng thái chờ đối soát.
2. Không gửi lại cùng ý định nếu chưa kiểm tra idempotency key.
3. Chạy reconcile vị thế/tiền mặt trước khi mở lại live.

## Khôi phục
- Chỉ mở lại live sau khi:
  - Kill-switch test pass.
  - Risk limits pass.
  - Sandbox path pass (nếu có).

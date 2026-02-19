# Playbook sự cố (Incident playbook)

## 1) Khi thấy lệnh kẹt ở SENT
1. Bật ngay **DỪNG KHẨN CẤP (Kill-switch)** để chặn mọi execute mới.
2. Mở `GET /oms/orders/{id}` và `GET /oms/audit` để xác nhận chuỗi trạng thái gần nhất.
3. Chạy đối soát: `POST /reconcile/run`.
4. Nếu lệnh chưa có ACK/FILL, giữ trạng thái theo audit; **không gửi lại** cùng ý định nếu chưa kiểm tra idempotency.

## 2) Khi mismatch fill
1. Chạy `GET /reconcile/latest` để xem `mismatches`.
2. So khớp từng `order_id` có trạng thái FILLED/PARTIAL_FILLED nhưng thiếu bản ghi fill.
3. Tạo bản ghi khắc phục thủ công có `correlation_id` rõ ràng (nếu cần) rồi chạy lại đối soát.
4. Chỉ resume khi `status=OK`.

## 3) Khi data stale
1. Kiểm tra `GET /healthz/detail` các trường `data_freshness_ok`, `as_of_date`.
2. Nếu stale: khóa execute (giữ Draft/Approve), nạp dữ liệu mới, xác nhận freshness trước khi mở lại.
3. Không bypass rule stale ở server-side.

## 4) Khi slippage anomaly
1. Tạm dừng execute bằng kill-switch.
2. Kiểm tra sự kiện `risk_blocked`, `execute_started`, `execute_completed` theo `correlation_id`.
3. So sánh giá dự kiến và giá fill trong `oms_fills`, đánh dấu phiên có bất thường.
4. Chỉ mở lại khi nguyên nhân được ghi rõ trong biên bản vận hành.

## 5) Cách bật kill-switch
- UI: dùng widget **DỪNG KHẨN CẤP (Kill-switch)**, tick xác nhận + gõ `DUNG` rồi bấm bật.
- API:
  - `POST /controls/kill_switch/on`
  - `POST /controls/kill_switch/off`
  - `POST /controls/pause` với `reason_code`
  - `POST /controls/resume`

> Lưu ý: Kill-switch chỉ chặn execute. Tạo lệnh nháp vẫn được phép để không mất luồng chuẩn bị.

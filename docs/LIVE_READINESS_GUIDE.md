# Hướng dẫn sẵn sàng giao dịch thật (Live Readiness Guide)

## Checklist bật live
- Bật `TRADING_ENV=live` và `ENABLE_LIVE_TRADING=true` một cách tường minh.
- Xác minh `KILL_SWITCH=false` trước khi mở phiên; thử bật/tắt kill-switch để chứng minh cơ chế chặn lệnh hoạt động.
- Chạy qua môi trường sandbox/testnet trước (nếu broker hỗ trợ).
- Kiểm tra giới hạn rủi ro: `RISK_MAX_NOTIONAL_PER_ORDER`, `RISK_MAX_ORDERS_PER_DAY`, giới hạn lỗ ngày.
- Đảm bảo audit log ghi đầy đủ chuyển trạng thái và mã chặn.

## Quy tắc an toàn
- Bắt đầu với giá trị lệnh nhỏ, tăng dần theo từng phiên.
- Luôn giám sát trạng thái hệ thống, không để live chạy không người theo dõi.
- Đây là tín hiệu nghiên cứu (Research signal), không phải lời khuyên đầu tư.
- Không cam kết lợi nhuận; luôn có rủi ro thua lỗ.

## Nhắc nhở pháp lý/độ tuổi
- Nếu dưới 18 tuổi có thể cần người giám hộ/đủ tuổi để dùng dịch vụ tài chính.
- Không hỗ trợ hoặc hướng dẫn lách luật.


## Xác minh trước khi READY
- Chạy `make verify-offline` (bắt buộc).
- Chạy `make verify-e2e` khi có Redis.
- Chạy `make verify-live-sandbox` để kiểm tra luồng place->ack->fill trên sandbox mock.
- Lưu evidence: python --version, git rev-parse HEAD, git status, log verify.

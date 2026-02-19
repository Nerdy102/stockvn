# Hướng dẫn Chế độ đơn giản (Simple Mode)

## 1) Dùng Chế độ đơn giản (Simple Mode) trong 3 bước
1. **Bước 1 — Chọn mã & chế độ**: nhập mã cổ phiếu (Symbol) như `FPT`, chọn khung thời gian (Timeframe) `1D` hoặc `60m`, rồi chọn chế độ chạy (Mode):
   - **Giao dịch giấy (Paper trading)**
   - **Lệnh nháp (Order draft)**
   - **Giao dịch thật (Live trading)** chỉ hiện khi bật cấu hình `ENABLE_LIVE_TRADING=true`.
2. **Bước 2 — Chọn mô hình & chạy**: chọn 1 trong 3 mô hình cố định trong Bộ mô hình (Model Zoo), bấm **Chạy phân tích (Run analysis)** để xem tín hiệu nghiên cứu (Research signal), độ tin cậy (Confidence), giải thích ngắn (Short explanation), rủi ro (Risks), biểu đồ tối giản (Minimal chart), và giả lập phí/thuế (Fee/Tax simulation).
3. **Bước 3 — Gợi ý lệnh & xác nhận**: hệ thống tạo Lệnh nháp (Order draft) có làm tròn lô chẵn (Board lot), làm tròn bước giá (Tick rounding), và ước tính phí/thuế/trượt giá; chỉ thực hiện khi người dùng bấm **XÁC NHẬN THỰC HIỆN (Confirm execute)**.

## 2) Đồng bộ dữ liệu (Data sync)
- **Demo offline (CSV/synthetic)**: luôn chạy được.
- **Plugin nhà cung cấp dữ liệu (Data provider plugin)**: người dùng tự cấu hình khóa API hợp lệ.
- Không crawl trái phép; không hardcode thông tin bí mật (secrets).

## 3) Bộ mô hình (Model Zoo) cố định
- **Mô hình 1 — Xu hướng (Trend-following)**: EMA20/EMA50 + breakout + volume.
- **Mô hình 2 — Hồi quy về trung bình (Mean-reversion)**: RSI14 + khoảng cách đến EMA20 + ATR%.
- **Mô hình 3 — Kết hợp nhân tố + chế độ thị trường (Factor + Regime)**: đa yếu tố + lọc risk-off, tần suất giao dịch thấp hơn.

## 4) So sánh mô hình (Model comparison)
- Bảng xếp hạng (Leaderboard): CAGR, MDD, Sharpe, turnover, lợi nhuận ròng sau phí/thuế (Net return after fees/taxes).
- Có tuỳ chọn **Xem chi tiết (Detailed)** để xem:
  - Giá trị danh mục theo thời gian (Equity curve)
  - Sụt giảm (Drawdown)
  - Danh sách giao dịch (Trade list) và tải CSV
- Có hash tái lập: config hash, dataset hash, code hash.
- **Cảnh báo lớn**: quá khứ không đảm bảo tương lai (Past performance is not indicative of future results); có rủi ro overfit; chi phí thực tế có thể khác mô phỏng.

## 5) Cảnh báo rủi ro & pháp lý
- Không phải lời khuyên đầu tư (Not investment advice).
- Có thể thua lỗ (Risk of loss).
- Dưới 18 tuổi (Under 18) cần tuân thủ điều kiện pháp lý; hệ thống không hỗ trợ lách luật.
- Giao dịch thật (Live trading) mặc định **TẮT**.

## 6) Kiểm tra hiển thị tiếng Việt có dấu
Simple Mode có banner kiểm tra hiển thị dấu:
> “Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư”.

Mục tiêu là phát hiện sớm lỗi font/thiếu glyph trên môi trường triển khai.


## 7) Đi từ dashboard sang Simple Mode
- Ở trang **🏠 Tổng quan hôm nay (Tổng quan hôm nay)**, bấm **Mở chế độ đơn giản (Open Simple Mode)** tại dòng tín hiệu.
- Hệ thống tự điền sẵn mã, khung thời gian (Timeframe), và mô hình vào wizard 3 bước.
- Sau đó bạn tạo **Lệnh nháp (Order draft)** và bấm xác nhận để ghi **Giao dịch giấy (Paper trading)** hoặc lưu nháp.

# Hướng dẫn Simple Mode (Dễ dùng cho người không IT)

## 1) Dùng Simple Mode trong 3 bước
1. **Bước 1**: Nhập mã cổ phiếu (ví dụ `FPT`), chọn timeframe (`1D` hoặc `60m`), chọn chế độ chạy (`paper` hoặc `draft`), bấm **Đồng bộ dữ liệu**.
2. **Bước 2**: Chọn một trong 3 model (Xu hướng / Hồi quy / Factor+Regime), bấm **Chạy phân tích** để xem tín hiệu nghiên cứu, độ tin cậy, giải thích ngắn và rủi ro.
3. **Bước 3**: Xem lệnh nháp đã tính sẵn phí/thuế/slippage, tick/lot; tích checkbox xác nhận rồi bấm **XÁC NHẬN THỰC HIỆN**.

## 2) Nạp dữ liệu: demo hoặc data_drop
- **Demo offline**: chạy ngay với dữ liệu `data_demo` (mặc định, luôn chạy được).
- **Data drop**:
  - Thả file CSV vào `data_drop/inbox/`.
  - Chạy:
    ```bash
    python -m scripts.ingest_data_drop --inbox data_drop/inbox --mapping configs/providers/data_drop_default.yaml --out data_demo/prices_demo_1d.csv
    ```
  - Hệ thống normalize dữ liệu để Simple Mode dùng ngay.

## 3) Giải thích ngắn 3 model
- **Model 1 — Xu hướng**: theo dõi EMA20/EMA50 và breakout + volume. Dùng khi muốn bám xu hướng.
- **Model 2 — Hồi quy về trung bình**: dùng RSI14 + khoảng cách đến EMA20 + ATR% để tìm điểm quá mua/quá bán ngắn hạn.
- **Model 3 — Factor + Regime**: kết hợp nhiều yếu tố và giảm giao dịch khi regime xấu (risk-off).

## 4) Cảnh báo rủi ro (bắt buộc đọc)
- Đây là **công cụ giáo dục**, **không phải lời khuyên đầu tư**.
- Hiệu quả quá khứ **không đảm bảo** kết quả tương lai.
- Bạn có thể thua lỗ do biến động thị trường, thanh khoản, trượt giá, phí/thuế.
- Live trading mặc định **TẮT**. Trong phạm vi hiện tại chỉ dùng paper trading và order draft.
- Nếu dưới 18 tuổi, có thể cần người giám hộ/đủ tuổi để mở tài khoản chứng khoán; hệ thống không hỗ trợ lách luật.

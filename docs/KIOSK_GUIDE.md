# KIOSK GUIDE — Mở web → bấm 2 nút → xong

## Mục tiêu
Kiosk là giao diện **siêu đơn giản** để người mới nhìn 10 giây là hiểu:
- Hôm nay thị trường ra sao
- Nên xem mã nào
- Mô hình có đang ổn không
- Tài khoản giấy đang lãi/lỗ thế nào

## Chạy nhanh offline (không cần Redis/Docker)
```bash
make setup
make run-api
make run-ui-kiosk
```

Mở trình duyệt:
- API docs: `http://localhost:8000/docs`
- Kiosk: `http://localhost:8502`

## Cách dùng trong 30 giây
1. Mở Kiosk, đọc khối **🏠 Hôm nay**.
2. Bấm nút lớn **Xem tín hiệu hôm nay** để mở danh sách gợi ý MUA/BÁN (nháp).
3. Bấm **Tạo lệnh nháp** ngay trên từng dòng hoặc nút lớn đầu trang.
4. Kiểm tra thông tin, chỉ xác nhận thủ công theo quy trình Draft → Confirm.

## Lưu ý an toàn
- Không có auto-trade.
- Live mặc định tắt.
- Nếu dưới 18 tuổi: chỉ dùng Draft/Paper.
- Câu nhắc bắt buộc: “Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư.”

## Bản tin + giải thích 1 câu
- Card **Hôm nay thị trường** dùng bản tin siêu ngắn, dễ hiểu cho người mới.
- Mỗi tín hiệu có **1 câu vì sao** và phần mở rộng giải thích thêm (ẩn mặc định).
- Phần so sánh mô hình có kiểu kể chuyện: ví dụ 10.000.000đ giả lập 1 năm qua.

# Quant Trust Pack

## Mục tiêu
Quant Trust Pack giúp đánh giá **độ tin cậy thống kê** của kết quả mô phỏng trước khi vận hành tiền thật.

## PSR/DSR là gì?
- **PSR (Probabilistic Sharpe Ratio):** xác suất Sharpe thực sự vượt một ngưỡng kỳ vọng.
- **DSR (Deflated Sharpe Ratio):** PSR đã điều chỉnh cho việc thử nhiều cấu hình, giảm rủi ro “chọn nhầm do may mắn”.

## MinTRL là gì?
- **MinTRL (Minimum Track Record Length):** số quan sát tối thiểu cần có để đạt mức tin cậy mục tiêu.
- Nếu Sharpe không vượt ngưỡng, MinTRL sẽ không khả dụng.

## Multiple testing disclosure
Càng thử nhiều mô hình/kịch bản, càng dễ có “winner giả”.
Vì vậy báo cáo luôn công bố `N_trials`, `N_eff`, `V_sr`, `SR0`.

## Bootstrap CI
Khoảng tin cậy bootstrap cho biết vùng bất định của chỉ số (ví dụ net return, Sharpe),
không phải cam kết lợi nhuận.

## Cảnh báo bắt buộc
- “Tin cậy thống kê” **không phải** cam kết lợi nhuận.
- Live trading mặc định tắt, chỉ mở khi đầy đủ điều kiện an toàn.

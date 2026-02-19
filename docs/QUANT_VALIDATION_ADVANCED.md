# Kiểm định định lượng nâng cao (Quant Validation Advanced)

## PBO/CSCV là gì?
- **PBO (Probability of Backtest Overfitting)** ước lượng rủi ro chọn nhầm mô hình do quá khớp dữ liệu.
- Đọc nhanh: **Thấp** (<=0.10), **Vừa** (0.10–0.25), **Cao** (>0.25).

## RC/SPA là gì?
- **RC (White Reality Check)** và **SPA (Hansen SPA)** giúp kiểm tra data snooping khi so nhiều trial.
- p-value chỉ là tham khảo thống kê, không đảm bảo lợi nhuận tương lai.

## HRP là gì?
- **HRP (Hierarchical Risk Parity)** phân bổ vốn dựa trên cấu trúc tương quan giữa các mã,
  giúp danh mục ổn định hơn ngoài mẫu.

## VaR/ES là gì?
- **VaR (Value at Risk)**: ngưỡng lỗ ở mức xác suất.
- **ES (Expected Shortfall)**: lỗ kỳ vọng trong phần đuôi xấu hơn VaR.

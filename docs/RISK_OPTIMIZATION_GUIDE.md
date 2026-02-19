# Risk optimization

- Position sizing cố định theo ATR và NAV.
- Kill-switch rules cho daily loss, drawdown, data stale.
- Walk-forward chấm điểm ổn định theo độ lệch chuẩn net return qua các fold.
- Monitoring drift: nếu trượt giá thực tế > 2x ước tính thì chuyển PAUSED.

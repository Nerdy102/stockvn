# VN10 EODHD Bootstrap and Backtest

## Mục tiêu
Tải dữ liệu EOD daily cho 10 mã VN, seed vào hệ thống demo, rồi chạy backtest đánh giá model người dùng theo chuẩn không look-ahead.

## Setup
```bash
make setup
```

## Add API token
Có 2 cách (ưu tiên env):

1) Env var:
```bash
export EODHD_API_TOKEN="YOUR_TOKEN"
```

2) Local secret file (đã ignore git):
```python
# configs/secrets_local.py
EODHD_API_TOKEN = "PASTE_TOKEN_HERE"
```

## Chạy pipeline
```bash
make bootstrap-vn10
make eval-vn10
```

## Assumptions quan trọng
- Tín hiệu tính tại EOD ngày `t`.
- Khớp lệnh tại OPEN ngày `t+1` (fallback CLOSE `t+1` nếu thiếu OPEN).
- Long-only, không leverage (gross <= 1).
- Chọn `top_k` mã có score > `min_score`, equal-weight.
- Có phí + slippage khi đổi vị thế:
  - fee mặc định 10 bps/chiều
  - slippage mặc định 5 bps/chiều

## Output
- `reports/vn10_backtest_report.md`
- `reports/vn10_equity_curve.csv`
- `reports/vn10_trades.csv`
- `reports/vn10_config.json`
- `reports/vn10_equity_curve.html`
- `reports/vn10_drawdown.html`

## Cảnh báo
- EODHD free plan thường giới hạn khoảng 1 năm lịch sử.
- Dữ liệu demo/indicative, **không dùng để trading live**.

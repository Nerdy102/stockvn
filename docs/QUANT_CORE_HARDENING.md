# Quant core hardening

## Nội dung đã siết chặt
- Cổng chất lượng dữ liệu trước khi phát tín hiệu (`data/quality/gates.py`).
- Không look-ahead cho breakout 20 phiên: dùng `high20_prev = rolling(20).max().shift(1)`.
- Chuẩn hoá regime/liquidity/confidence/risk tags theo quy tắc cố định.
- Bổ sung `debug_fields` để audit, ẩn khỏi UI.

## Mặc định ngưỡng
- 1D tối thiểu 120 phiên.
- 60m tối thiểu 300 phiên.
- Tỷ lệ thiếu dữ liệu tối đa 2%.

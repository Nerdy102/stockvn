# Hướng dẫn Corporate Actions

## Cách nạp dữ liệu
Thả CSV vào `data_drop/corporate_actions/` với cột:
`symbol, ex_date, action_type, amount, currency, note`

## Cảnh báo adjusted/unadjusted
- Dữ liệu adjusted: không apply split lần nữa để tránh double-adjust.
- Dữ liệu unadjusted: apply split theo cấu hình.

# Hướng dẫn Universe Snapshot

## Vì sao cần membership history?
Để tránh survivorship bias, mỗi backtest phải dùng danh sách thành viên tại thời điểm lịch sử.

## Cách nạp file
Thả CSV vào `data_drop/universe_snapshots/` với cột:
`snapshot_date, universe_name, symbols` (symbols ngăn cách bằng dấu phẩy).

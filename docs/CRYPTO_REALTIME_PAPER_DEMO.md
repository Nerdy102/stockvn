# Crypto Realtime Paper Demo (Binance BTCUSDT)

Mục tiêu: ingest trade realtime BTCUSDT (public Binance), build bar 15m/60m, phát signal + realtime summary trong Redis, và demo OMS paper execute theo giá realtime.

## 0) Prerequisites
- Redis chạy tại `redis://localhost:6379/0`
- API chạy (`make run-api`)
- `REALTIME_ENABLED=true` trong `.env`

## 1) Start Redis (optional helper)
```bash
make redis
```

## 2) Start components (mỗi lệnh ở 1 terminal)
```bash
make rt-btc-ingest
make rt-btc-bars
make rt-btc-signals
```

Các lệnh tương đương:
```bash
python -m services.stream_ingestor.binance_trade_ingestor --symbols BTCUSDT --redis redis://localhost:6379/0
python -m services.bar_builder.run --redis redis://localhost:6379/0 --exchange CRYPTO
python -m services.realtime_signal_engine.run --redis redis://localhost:6379/0
```

## 3) Verify realtime API
```bash
curl -s 'http://localhost:8000/realtime/summary' | jq .
curl -s 'http://localhost:8000/realtime/bars?symbol=BTCUSDT&tf=15m&limit=3' | jq .
```

## 4) Demo paper OMS execute using realtime price
```bash
make rt-btc-demo-order
```
Lệnh này đọc close mới nhất từ `realtime:bars:BTCUSDT:15m`, tạo draft -> approve -> execute (paper/sandbox, market=crypto).

## Notes / Safety
- Đây là demo paper/sandbox, **không live trading**.
- Không cần API key broker.
- Binance stream dùng public websocket trade stream:
  - `wss://stream.binance.com:9443/ws/<symbol>@trade`
  - Combined streams: `wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade`

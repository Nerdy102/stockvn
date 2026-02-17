# SSI FastConnect Field Mapping

## Alias normalization
- `TradingDate`/`Tradingdate` -> trading_date
- `MarketID`/`MarketId` -> market_id
- `Avg`/`AvgPrice`/`AveragePrice` -> avg_price
- `Ceiling`/`CeilingPrice` -> ceiling_price
- `Floor`/`FloorPrice` -> floor_price
- `Ref`/`RefPrice` -> ref_price
- `PriorVal` -> prior_close
- `TradingTime` is accepted as alias of `Time`.

## Timezone handling
- Parse `dd/MM/yyyy` + `HH:MM:SS` or `HHMMSS` in `Asia/Ho_Chi_Minh`.
- Convert to UTC and store as canonical `ts_utc`.

## Canonical mapping summary
- `DailyOhlc` / `IntradayOhlc` -> `Bar`
- `DailyStockPrice` -> `Bar` + market daily meta (`ref/ceiling/floor`)
- Streaming `X-QUOTE` -> `QuoteLevel`
- Streaming `X-TRADE` -> `TradePrint`
- Streaming `X` -> `(QuoteLevel, TradePrint)`
- Streaming `R` -> `ForeignRoomSnapshot`
- Streaming `MI` -> `IndexSnapshot`
- Streaming `B` -> `Bar` timeframe `1m`
- Streaming `OL` -> `OddLotSnapshot`

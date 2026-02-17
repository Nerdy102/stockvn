# SSI Contract

This project supports an offline-first SSI FastConnect contract layer with:

- Raw tolerant parsing (`ssi_raw_models.py`) with alias/typo mapping.
- Canonical typed models (`data/schemas/canonical.py`).
- Deterministic mapper entrypoint (`providers/ssi_fastconnect/mapper.py`).

## Alias mapping

- `MarketID`, `MarketId` -> `market_id`
- `Isin`, `ISIN` -> `isin`
- `TotalMatchVol`, `Totalmatchvol`, `TotalMatchVOL` -> `total_match_vol`
- `TotalMatchVal`, `Totalmatchval` -> `total_match_val`
- `ForeignSellValTotal`, `Toreignsellvaltotal` -> `foreign_sell_val_total`
- `NetForeiVol`, `Netforeivol` -> `net_foreign_vol`
- `Tradingdate`, `TradingDate` -> `trading_date`

## Timezone

Input SSI values are interpreted as `Asia/Ho_Chi_Minh` and mapped to UTC timestamps.

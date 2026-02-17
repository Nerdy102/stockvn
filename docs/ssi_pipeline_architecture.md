# SSI-standardized Bronze/Silver/Gold pipeline

## Canonical contract
- SSI REST/streaming schemas are defined in `packages/data/data/schemas/ssi_fcdata.py`.
- Unknown fields are allowed (`extra=allow`) and counted in `schema_unknown_fields_total` for drift monitoring.
- Typo aliases such as `Toreignsellvaltotal`, `Netforeivol` are mapped.

## Data layers
- **Bronze**: `BronzeRaw` append-only payload archive with dedup hash `(provider_name, endpoint_or_channel, payload_hash)`.
- **Silver**: normalized tables (`PriceOHLCV`, `QuoteL2`, `TradeTape`, `ForeignRoom`, `IndexOHLCV`) with idempotent upsert keys.
- **Gold**: indicator/factor/signal outputs and `IndicatorState` for incremental computation.

## Worker flow
1. Ingest raw provider payloads into Bronze.
2. Map provider records via adapters (`SSIMapper`) to canonical DTOs.
3. Upsert Silver and update `IngestState` checkpoint.
4. Run incremental compute jobs (`compute_indicators_incremental`) using `IndicatorState`.

## Scale notes
- Indexes added on high-frequency predicates (`symbol,timeframe,timestamp`, factor date index).
- API endpoints require pagination / bounded date defaults to avoid full-table scans.
- Smoke tests in `tests/test_performance_smoke.py` provide CI-safe performance regression guard.

## Provider extension
- Add new provider fetcher to `data/providers/*`.
- Implement adapter mapping provider payload -> canonical DTO.
- Reuse pipeline repositories (`append_bronze`, `upsert_silver_price`, `update_ingest_state`).

## Local performance smoke
```bash
PYTHONPATH=packages/core:packages/data:services/api_fastapi:services/worker_scheduler pytest -q tests/test_performance_smoke.py
```

# Evaluation Lab Status

## Mandatory inventory
- `git rev-parse HEAD`: `b1753784f1fa7d6a77ed2ef3f4ce8a8f3be8a7ec`
- `git status --porcelain`: clean before this evaluation-lab patch series
- `git diff --stat`: empty before this evaluation-lab patch series

## Current backtest entrypoints
- `scripts/run_walkforward.py`
- `scripts/ml_walkforward.py`
- `scripts/run_raocmoe_backtest.py`
- `packages/core/core/ml/backtest.py`
- `packages/core/core/backtest_v2/engine.py`

## Existing anti-overfit / trust-pack test locations
- `tests/test_purged_kfold_no_leakage.py`
- `tests/test_rank_pairwise.py` (purged CV with embargo validation)
- `tests/test_cscv_pbo_sanity.py`
- `tests/test_quant_dsr_sr0_behavior.py`
- `packages/core/core/quant_validation_advanced/rc_spa.py` (Reality Check / SPA implementation)

## Current local dataset (data_demo)
- `data_demo/prices_demo_1d.csv`
  - schema: `date,symbol,open,high,low,close,volume,value_vnd`
  - min/max date: `2025-03-03` -> `2026-02-13`
  - rows: `2750`
- `data_demo/crypto_prices_demo_1d.csv`
  - schema: `symbol,date,open,high,low,close,volume,value_vnd`
  - min/max date: `2024-01-01` -> `2024-09-16`
  - rows: `1300`

## Realtime Redis storage check
Command used: `rg -n "_bar_cache|_kv|_hot|XRANGE|XREAD|cursor:" services/`
Findings:
- `services/bar_builder/bar_builder/storage.py` still contains `_bar_cache` fallback path for non-redis test doubles.
- `services/realtime_signal_engine/realtime_signal_engine/state_store.py` still contains `_kv/_hot` fallback paths for non-redis test doubles.
- `services/realtime_signal_engine/realtime_signal_engine/engine.py` uses stream cursor key `cursor:bar_close:{tf}` and `xread` path; fallback `xrange` is present for compatibility.
- `services/bar_builder/bar_builder/consumer.py` uses cursor key `cursor:market_events` with `xread`.

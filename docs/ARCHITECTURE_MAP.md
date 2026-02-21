# ARCHITECTURE MAP

## Services
- `services/api_fastapi`: REST APIs for app/dashboard and run orchestration.
- `services/worker_scheduler`: scheduled workers and async task execution.
- `services/stream_ingestor`: realtime ingestion pipeline.
- `services/bar_builder`: bar aggregation for realtime feeds.
- `services/realtime_signal_engine`: signal generation from live bars.

## Core Packages
- `packages/core/core/eval_lab/*`: metrics, multiple testing (RC/SPA/PBO/PSR/DSR), splits/bootstrap/consistency.
- `research/strategies/*`: baseline + USER/RAOCMOE strategy weight generators.

## Data Flow
1. Dataset CSV (`data_demo/*`) loaded by `scripts/run_eval_lab.py`.
2. Strategy weights generated per decision date.
3. T+1 simulation computes equity, turnover, costs, and consistency checks.
4. Eval artifacts written under `reports/eval_lab/{run_id}/`:
   - `results_table.csv`, `summary.json|md`, `stats/*`, `equity_curves/*`, `model_outputs/*`.
5. Chat scripts read latest run artifacts and print scoreboard/snapshot.

## Redis Keys / Realtime (current)
- Realtime pipeline uses `REDIS_URL` and service defaults; key names are managed inside realtime services and harness tooling.
- Cursor-based replay/hot-cache invariants are covered by tests in `tests/test_redis_hot_cache_rolls_last_200.py` and replay tests.

## UI Pages
- Streamlit dashboards under `apps/dashboard_streamlit` and `apps/web_kiosk`.
- Interactive console status tracked in `docs/INTERACTIVE_CONSOLE_STATUS.md`.

# RAOCMOE Status

## Repo status report
- `git rev-parse HEAD`: `80040f07387db09d1d64be807bd78fd9c54026aa`
- `git status --porcelain`: clean before this patch series
- `git diff --stat`: empty before this patch series

## Existing quant pipeline entrypoints
- Scripts: `scripts/ml_walkforward.py`, `scripts/run_walkforward.py`, `scripts/alpha_v3_train.py`, `scripts/alpha_v3_predict.py`, `scripts/run_sensitivity.py`.
- Worker jobs in `services/worker_scheduler/worker_scheduler/jobs.py`: `job_train_alpha_v3`, `job_update_alpha_v3_cp`, `run_overfit_controls_alpha_v3`, plus replay/bronze/silver/gold and monitoring jobs.

## Existing realtime architecture and key conventions
- `services/bar_builder`: writes list cache key `realtime:bars:{symbol}:{tf}` and emits `stream:bar_close:{tf}`.
- `services/realtime_signal_engine`: consumes `stream:bar_close:{tf}` using cursor key `cursor:bar_close:{tf}`.
- `services/realtime_signal_engine/state_store.py` writes:
  - `realtime:state:ind:{symbol}:{tf}`
  - `realtime:signals:{symbol}:{tf}`
  - `realtime:ops:summary`
  - `realtime:hot:{name}`
- `services/api_fastapi/api_fastapi/routers/realtime.py` reads realtime state through Redis `GET` and `LRANGE`.

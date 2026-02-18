# VERIFICATION REPORT

## 1) Summary

| Layer | Status | Notes |
|---|---|---|
| T0 Static quality gate | FAIL | `make quality-gate` failed on pre-existing repo-wide lint debt (157 ruff violations). UI guardrail passed. |
| T1 Unit tests | PASS | Full pytest passed: 253 passed, 4 skipped. |
| T2 Integration | PARTIAL | Seed/jobs/endpoints executed offline with evidence; several endpoint contracts not available or returned validation errors in current build. |
| T3 E2E smoke | PASS | OpenAPI JSON, health, and UI import smoke artifacts generated. |
| T4 Parity | PASS | backtest/replay/paper parity holds with zero diffs. |

## 2) Bugs found + fixes
- **/ml/diagnostics returned only insufficient_data without canonical metric keys** → Populate zero-valued diagnostics metrics even on insufficient data path (`services/api_fastapi/api_fastapi/routers/ml.py`)
- **job_train_alpha_v3 reported predictions including CP side-effects leading to mismatch** → Return persisted alpha prediction count only (`services/worker_scheduler/worker_scheduler/jobs.py`)
- **Alembic migration 0005 failed when event_log table already existed** → Make migration idempotent by guarding table/index create/drop (`migrations/versions/20260218_0005_event_log.py`)
- **compute_factors crashed on duplicate (symbol,date) rows during reseed** → Deduplicate prices by (symbol,date) before pivot and add regression test (`packages/core/core/factors.py, tests/test_factors_dedup_symbol_date.py`)
- **PIT fundamentals merge failed with null keys** → Drop null symbol/effective_public_date rows before asof merge and add regression test (`packages/core/core/alpha_v3/features.py, tests/test_alpha_v3_features_drop_null_pit_keys.py`)

## 3) Golden/oracle changes
- No golden snapshot files changed. Added regression unit tests only.

## 4) Perf summary
- Feature build (200x500): {'seconds': 0.731, 'peak_mem_mb': 16.54, 'rows_scores': 200}
- Screener run: {'seconds': 0.029, 'status_code': 422}
- Chart downsample: {'input_points': 20000, 'output_points': 10000}

## 5) Known limitations
- T0 cannot pass without repo-wide lint normalization outside minimal hardening scope.
- Some requested endpoints (`/prices/latest_date`, `/watchlists`, `/alerts`, `/reports/export`) are absent or return non-2xx in this build; captured as evidence.
- SMTP/runtime external integrations remain disabled without env credentials (expected offline behavior).

## 6) Next steps (CI replay)
1. `make quality-gate`
2. `python scripts/ui_guardrail_check.py`
3. `PYTHONPATH=.:packages/core:packages/data:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps pytest -q`
4. `python scripts/seed_demo_data.py`
5. Re-run integration harness used in this verification to refresh artifacts under `artifacts/verification/`.
6. `python -m scripts.replay_smoke`

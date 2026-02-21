# TEST TRIAGE

## Pre-existing Failure Snapshot (full suite with PYTHONPATH)
Command:
`PYTHONPATH=.:services:packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps pytest -q --maxfail=5`

Top 5 failures observed (pre-existing):
1. `tests/test_no_live_when_disabled.py::test_no_live_when_disabled`
   - expected `LIVE_DISABLED`, got `SANDBOX_REQUIRED`.
2. `tests/test_oms_state_machine_idempotency.py::test_confirm_execute_idempotent_double_click`
   - expected status 200, got 422.
3. `tests/test_oms_transition_sequence.py::test_paper_order_writes_oms_sequence_to_audit_log`
   - expected status 200, got 422.
4. `tests/test_simple_dashboard_determinism_hash.py::test_dashboard_determinism_hash`
   - fails due backend/db state assumptions.
5. `tests/test_simple_dashboard_payload_smoke.py::test_simple_dashboard_payload_smoke`
   - fails with DB table setup missing (`universe_snapshots`).

## Current Cycle
- New/updated tests for improvement artifacts and chat outputs pass.
- Pre-existing failure count not increased in this cycle.

## Notes
- Full suite run: `5 failed, 414 passed, 3 skipped` with `--maxfail=5` stopping early after top failures.

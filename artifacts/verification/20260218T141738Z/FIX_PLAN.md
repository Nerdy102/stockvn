# FIX PLAN

## Root cause hypothesis
1. Quality gate mypy failure: module discovered under both `realtime_harness.*` and `tools.realtime_harness.*` due path/package layout.
2. Offline batch worker failure: sqlite locking during sequential job execution (`compute_data_quality_metrics_job`) under same run.
3. Full-suite test instability: environment import-path/runtime constraints and long-running suite not completed in one pass.

## Minimal patch plan
1. Normalize package import roots for realtime harness (single canonical module path) and align mypy invocation.
2. Add bounded retry/backoff or session boundary isolation for `JobRun` writes in worker jobs to prevent sqlite lock errors in local/offline mode.
3. Add deterministic CI command wrappers for full pytest with explicit PYTHONPATH and split shards if needed.

## Tests to add
1. Regression test for worker `--once` completing all scheduled jobs without sqlite lock using temporary sqlite DB.
2. Static test asserting `make quality-gate` mypy invocation succeeds with canonical module paths.
3. End-to-end offline demo smoke test that asserts non-error exit and idempotent row deltas.

## Rollback flags
1. Keep realtime disabled by default (`REALTIME_ENABLED=false`) until T0/T1/T2 pass.
2. Use offline replay-only mode for demo verification until sqlite lock fix lands.

# FIX_PLAN

## Root cause hypothesis
1. Full-suite `pytest -q` contains at least one long-running/hanging path in current environment.
2. Offline worker run invokes redis-backed stream job paths and fails without redis service.
3. Realtime smoke requires docker-managed services; docker CLI/runtime unavailable in this environment.

## Minimal patch plan
1. Isolate failing/hanging pytest subset with shard strategy and add timeout guards for long integration tests.
2. Gate redis-dependent jobs in `--once` offline mode behind availability checks or explicit flags.
3. Add fallback local orchestration docs/scripts for environments without docker.

## Tests to add
1. Worker `--once` integration test with redis unavailable should fail gracefully with clear diagnostics.
2. Regression test for bounded-time completion of known long integration test paths.
3. Realtime smoke contract test using fake redis stream harness for environments without docker.

## Rollback flags
1. Keep realtime flags default OFF.
2. Disable order generation when governance/reconcile/SLO incidents are unresolved.

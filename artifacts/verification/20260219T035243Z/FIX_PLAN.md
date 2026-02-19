# FIX PLAN

## Root cause hypotheses
1. Full test suite runtime exceeded practical local execution window; potential slow/integration-heavy tests need CI runtime with extended limits.
2. Some requested OMS/reconcile test names were unavailable or mismatched, causing partial evidence gaps for section T6.
3. Realtime full-stack service metrics were not collected from a dockerized service mesh in this run.

## Minimal patch plan
1. Add/align test entrypoints for OMS pretrade/reconcile/governance pause scenarios under stable names.
2. Add a dedicated `make absolute-verify` target to orchestrate all T0..T12 checks and artifact emission.
3. Add bounded timeout + progress logging for long-running pytest runs and generate junit XML.

## Tests to add
1. Integration test: replay twice in isolated redis namespaces with strict bar/signal hash equality assertion.
2. Integration test: reconciliation mismatch triggers governance PAUSE and blocks order generation while preserving analysis paths.
3. E2E API test: realtime disabled fallback payload contracts for all `/realtime/*` endpoints.

## Rollback flags
- Keep default-safe flags OFF in `.env.example` (`ENABLE_REALTIME_PIPELINE=false`, `REALTIME_*_ENABLED=false`).
- Use `GOVERNANCE_FORCE_PAUSE=true` as emergency circuit-breaker during rollout.

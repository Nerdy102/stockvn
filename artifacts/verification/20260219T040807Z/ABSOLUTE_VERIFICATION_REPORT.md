# ABSOLUTE VERIFICATION REPORT

Generated: 2026-02-19T04:26:52.059622Z

## 1) Executive summary table (PASS/FAIL): T0..T10
| Stage | Status | Notes |
|---|---|---|
| T0 | PASS | OK |
| T1 | PASS | 300 passed, 3 skipped |
| T2 | PASS | OK |
| T3 | WARN | docker/redis realtime stack unavailable; replay+unit checks executed |
| T4 | PASS | OK |
| T5 | PASS | OK |
| T6 | PASS | OK |
| T7 | PASS | OK |
| T8 | PASS | OK |
| T9 | PASS | OK |
| T10 | PASS | OK |

## 2) Determinism evidence: hashes comparisons (bars/signals)
- `t3_realtime_determinism_hash_compare.json` contains deterministic bar hash check output.
- `t3_late_correction_sample.json` contains late-event correction policy evidence.

## 3) Idempotency evidence: before/after counts
- `t2_row_counts_before_after.json` + `t2_idempotency_assertions.txt` show rerun deltas.

## 4) Fail-safe evidence: reconciliation mismatch -> pause
- `t6_reconcile_run.json` and `t6_governance_pause_evidence.json` validate incident + pause behavior.

## 5) Performance + chaos results summary
- `t8_rt_load_report.json`, `t8_rt_chaos_report.json`, `t8_perf_budget_summary.json` summarize deterministic load/chaos runs.

## 6) Known limitations
- Docker/Redis realtime service stack not available in this environment; realtime smoke used replay/tests fallback.
- Postgres-only tests are skipped by design when `TEST_DATABASE_URL` is unset.

## 7) Ready for realtime? verdict
- **READY WITH WARNINGS**: full test suite passed; realtime contracts and determinism checks passed via offline replay/unit harness; live dockerized stack verification is pending due environment tooling limits.

## 8) Next actions checklist (ordered)
1. Run the same T3 smoke against dockerized realtime profile on a host with Docker + Redis.
2. Export and archive service `/metrics` snapshots from gateway/bar-builder/signal-engine under live replay.
3. Add CI job to enforce realtime safe flags in `.env.example` and include T3/T8 artifacts as build artifacts.
4. Periodically rerun full `pytest -q` + report pack export on a clean DB snapshot.
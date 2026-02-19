# ABSOLUTE VERIFICATION REPORT

Generated: 2026-02-19T04:59:27.919163Z

## 1) Executive summary table (PASS/FAIL): T0..T10
| Stage | Status | Notes |
|---|---|---|
| T0 | PASS | OK |
| T1 | PASS | 300 passed, 3 skipped |
| T2 | PASS | OK |
| T3 | WARN | docker/redis not available; realtime verified via replay/unit fallback artifacts |
| T4 | PASS | OK |
| T5 | PASS | OK |
| T6 | PASS | OK |
| T7 | PASS | OK |
| T8 | PASS | OK |
| T9 | PASS | OK |
| T10 | PASS | OK |

## 2) Determinism evidence: hashes comparisons (bars/signals)
- `t3_realtime_determinism_hash_compare.json` (bar hash determinism).
- `t3_late_correction_sample.json` (late correction policy).

## 3) Idempotency evidence: before/after counts
- `t2_row_counts_before_after.json` + `t2_idempotency_assertions.txt`.

## 4) Fail-safe evidence: reconciliation mismatch -> pause
- `t6_reconcile_run.json` + `t6_governance_pause_evidence.json`.

## 5) Performance + chaos results summary
- `t8_rt_load_report.json`, `t8_rt_chaos_report.json`, `t8_perf_budget_summary.json`.

## 6) Known limitations
- Docker + Redis runtime not available in this environment for full realtime service orchestration.
- Postgres-only tests skipped when `TEST_DATABASE_URL` is unset.

## 7) Ready for realtime? verdict
- **READY WITH WARNINGS**: all mandatory tests passed; realtime live-stack remains an environment-limited warning with fallback deterministic evidence captured.

## 8) Next actions checklist (ordered)
1. Run T3 live stack checks on host with Docker and Redis enabled.
2. Capture `/metrics` snapshots from gateway/bar_builder/signal_engine under replay load.
3. Keep safe defaults in `.env.example` and assert in CI.
4. Archive this evidence directory with release tag.
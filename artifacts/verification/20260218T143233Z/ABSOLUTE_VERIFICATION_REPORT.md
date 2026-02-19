# ABSOLUTE VERIFICATION REPORT

## 1) Executive summary table (PASS/FAIL): T0..T10
| Stage | Status | Evidence |
|---|---|---|
| T0 | PASS | `t0_quality_gate.txt`, `t0_ui_guardrails.txt` |
| T1 | FAIL | `t1_pytest.txt` (`pytest -q` timed out before completion) |
| T2 | FAIL | `t2_batch_pipeline_log.txt` (worker `--once` failed due Redis connection) |
| T3 | FAIL | `t3_realtime_smoke_log.txt` (`docker` unavailable + Redis refused) |
| T4 | PASS | `t4_realtime_api_samples.json`, `t4_realtime_api_bounds.txt`, `t4_realtime_api_disabled_samples.json` |
| T5 | PASS | `t5_ui_import_smoke.txt`, `t5_ui_budget_checks.txt`, `t5_ui_realtime_toggle_behavior.txt` |
| T6 | PASS | `t6_oms_risk_tests.txt` + `t6_*` evidence json |
| T7 | PASS | `t7_observability_tests.txt`, `t7_incidents_created.json`, `t7_runbook_links_check.txt` |
| T8 | PASS | `t8_rt_load_report.json`, `t8_rt_chaos_report.json`, `t8_perf_budget_summary.json` |
| T9 | PASS | `t9_quant_tests.txt`, `t9_*` monitoring json |
| T10 | PASS | `t10_report_tests.txt`, `t10_report_manifest.json`, `t10_report_sections_check.txt` |

## 2) Determinism evidence: hashes comparisons (bars/signals)
- Repeated load harness run with same seed/symbol/day produced identical `bars_hash`.
- `run1_bars_hash=1438a741da10a6115a0e7ca0dff388a473abad2cf4b4419ae5dd28bbc79325d9`
- `run2_bars_hash=1438a741da10a6115a0e7ca0dff388a473abad2cf4b4419ae5dd28bbc79325d9`
- `hash_match=True`
- Signal idempotency flag was true in both runs.

## 3) Idempotency evidence: before/after counts
- `t2_row_counts_before_after.json` captures row counts before/after rerun.
- `t2_idempotency_assertions.txt` shows zero deltas on tracked key tables.

## 4) Fail-safe evidence: reconciliation mismatch -> pause
- Verified via tests in `t6_oms_risk_tests.txt` (`test_reconcile_mismatch_creates_incident.py`, `test_governance_v3.py`) and summarized in `t6_governance_pause_evidence.json`.

## 5) Performance + chaos results summary
- Deterministic load and chaos harness reports generated: `t8_rt_load_report.json`, `t8_rt_chaos_report.json`, `t8_rt_load_report_full.json`.
- Budget/invariant rollup captured in `t8_perf_budget_summary.json`.

## 6) Known limitations (honest, concrete)
1. `pytest -q` full suite did not complete within timeout window (T1 fail).
2. Offline worker `--once` failed in this environment because Redis is not reachable on localhost:6379 during stream-related job execution.
3. Realtime smoke prerequisites could not be fully met because `docker` CLI is unavailable in container.

## 7) Ready for realtime? verdict
**NOT READY**

Reasons:
1. T1 full-unit gate not completed/passing.
2. T2 offline pipeline once-run failed in current environment.
3. T3 realtime smoke cannot be fully executed due missing docker/redis runtime dependencies.

## 8) Next actions checklist (ordered)
1. Stabilize/segment full test run to identify and fix the blocking tests causing timeout.
2. Provide reachable Redis (and/or mock path) for `worker_scheduler.main --once` offline verification to complete.
3. Provide docker runtime (or documented service startup alternative) to execute realtime smoke end-to-end.
4. Re-run T1/T2/T3 and regenerate full evidence pack.
5. Promote verdict to READY WITH WARNINGS or READY only after T1/T2/T3 are green.

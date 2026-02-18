# VN Invest Toolkit Verification Report

Generated: 2026-02-18T12:02:52.146664Z

## 1) Summary Table

| Layer | Status | Evidence |
|---|---|---|
| T0 Static quality gate | FAIL (ruff import-order baseline violations) | `t0_quality_gate.txt`, `t0_ui_guardrails.txt` |
| T1 Unit tests | PASS | `t1_pytest.txt`, `t1_failures.txt` |
| T2 Integration (API+DB+worker/idempotency) | PASS with noted missing endpoints | `t2_integration_log.txt`, `t2_row_counts.json` |
| T3 E2E smoke (API docs + UI imports + report pack) | PASS | `t3_api_health.json`, `t3_openapi_schema.json`, `t3_ui_import_ok.txt`, `t3_report_pack_manifest.json` |
| T4 Parity (backtest vs replay/paper) | PASS | `t4_parity_backtest.json`, `t4_parity_paper.json`, `t4_parity_diff.json`, `t4_parity_pass.txt` |

## 2) Bugs Found + Fixes
- No code-level bug fix committed in this verification run.
- Observed baseline issue: `make quality-gate` fails on pre-existing import sort (`I001`) violations; not changed to avoid broad non-functional churn.
- Observed API parity gap vs requested checklist: `/prices/latest_date`, `/alerts`, and `/reports/export` are absent in current router surface (captured as 404 in integration log).

## 3) Golden Changes
- None.

## 4) Perf Summary
- See `perf_summary.json` for feature build micro-run, screener response timing/status, and chart downsample 10k cap evidence.

## 5) Known Limitations
- Full T0 pass is blocked by existing lint debt unrelated to this run's verification scope.
- Some checklist endpoints are not implemented in this codebase; verification recorded actual behavior without altering business rules.
- SSI live provider disabled in DEV mode without credentials (expected offline behavior).

## 6) Next Steps (CI Repeatability)
1. `make quality-gate` and `python scripts/ui_guardrail_check.py`.
2. `PYTHONPATH=.:packages/core:packages/data:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps pytest -q`.
3. Run integration harness to refresh `t2_*`, `t3_*`, `t4_*`, and `perf_summary.json` artifacts.
4. Publish `artifacts/verification/*` as CI artifacts.

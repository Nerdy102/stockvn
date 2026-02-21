# AUTOPILOT STATUS

## Latest Run
- run_id: `6f7a35f285f029af` (eval lab)
- dataset: `vn_daily`
- verdict: `FAIL`
- reasons: `INSUFFICIENT_TRACK_RECORD`, `stress_fragility_high`, `sharpe_sign_sanity`
- chosen_default: `USER_V1_STABILITY`
- why_chosen: `fallback to best objective among USER variants (no PASS variant)`

## Mandatory Step 0 Inventory
- `git rev-parse HEAD`: `9600cdd42203e08bd26aec9c2efc9c079dfabd4f`
- `git status --porcelain`: clean at start
- `git diff --stat`: no diffs at start

## Ship Check Outputs
- `make quality-gate`: PASS
- `make eval-chat`: PASS
- `make model-chat`: PASS
- `python scripts/run_improvement_suite.py --dataset vn_daily`: PASS execution; `DEV_WINNER=NONE`, `FINAL_DEFAULT=NOT_VERIFIED`.

## Lockbox Protocol Snapshot
- DEV window: `2025-03-03..2025-12-03`
- LOCKBOX window: `2025-12-05..2026-02-11`
- DEV winner: `NONE` (no variant passed DEV reliability gate)
- LOCKBOX verdict for DEV winner: `NOT VERIFIED`
- Final default: `NOT_VERIFIED`

## Reliability Snapshot
- Data audit: PASS
- Identity abs_err: `0.000000e+00`
- PBO: `0.062500` (full eval run)
- RC/SPA: both `< 0.000500`
- DSR (`USER_V0`): `0.000000`
- Test days: `75`
- MinTRL (`USER_V0`): `1000000000`

## Fixes in this iteration
- Full-suite triage snapshot captured in `docs/TEST_TRIAGE.md` with top 5 pre-existing failures.
- Improvement suite output now uses deterministic objective helper functions and NA-safe formatter for `CostDrag_traded`.
- Added unit tests for objective formula and DEV winner selection logic.
- Improvement suite now exports `dev_scoreboard.csv` and `lockbox_scoreboard.csv` for UI consumption; eval-chat prints LOCKBOX VERIFY SUMMARY block when present.
- Improvement summary now stores artifact paths and both scoreboards include `objective_score` for DEV and LOCKBOX slices.

## Known Issues
- No USER variant is reliability PASS on DEV+LOCKBOX protocol yet.
- Need USER_V4..V7 additions and lockbox-verified selection before promoting default.

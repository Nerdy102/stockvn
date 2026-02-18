# FIX PLAN

## Root-cause hypotheses
- Environment and regression failures surfaced in T0/T1/T5/T4 enabled path.

## Minimal patch plan
- Address lint/type/test breakages from quality gate and full pytest failures.
- Add redis connection fallback in realtime enabled API paths.

## Tests to add
- Regression for realtime enabled mode without redis -> graceful payload.

## Rollback flags
- Keep realtime disabled via feature flags until green.

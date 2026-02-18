# ADR 0005: Fail-safe Governance (Pause-first)

- Status: Accepted
- Date: 2026-02-18

## Context
Realtime systems must fail safely under drift, coverage loss, or SLO breaches.

## Decision
When governance checks fail, system enters PAUSE mode:
- Disable order generation/submission.
- Keep analysis, monitoring, and alerts operational.
- Require explicit operator action for resume/rollback.

## Consequences
- Safety and auditability are prioritized over liveness.
- Realtime faults do not block offline/batch analytical surfaces.

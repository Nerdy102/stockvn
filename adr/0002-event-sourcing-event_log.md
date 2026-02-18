# ADR 0002: Event Sourcing with Append-only event_log

- Status: Accepted
- Date: 2026-02-18

## Context
Auditability and deterministic replay require immutable event history.

## Decision
Persist all canonical events in an append-only `event_log` with canonical JSON and SHA-256 hash.

## Consequences
- Reprocessing can be proven deterministic by hash parity checks.
- Incident/debug workflows can reconstruct full timelines without mutable-state ambiguity.

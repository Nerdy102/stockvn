# ADR 0004: Determinism & Hashing Standard

- Status: Accepted
- Date: 2026-02-18

## Context
Production verification requires deterministic outputs for replay, backtest, and realtime parity.

## Decision
All canonical contracts require:
- `schema_version`
- canonical JSON serialization (sorted keys, compact separators)
- SHA-256 `payload_hash`

Program-level verification combines dataset/config/code hashes for traceability.

## Consequences
- Same inputs produce identical hashes and reproducible outputs.
- Regression triage can identify whether drift came from code, config, or data.

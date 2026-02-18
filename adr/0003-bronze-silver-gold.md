# ADR 0003: Bronze/Silver/Gold Data Layers

- Status: Accepted
- Date: 2026-02-18

## Context
Provider payloads evolve frequently and cannot be coupled directly to analytics consumers.

## Decision
Adopt three-layer lakehouse model:
- Bronze: raw payload + envelope, append-only.
- Silver: normalized canonical market data with lineage.
- Gold: analytics-ready bars/features and downstream snapshots.

## Consequences
- Schema evolution is isolated in Bronze→Silver normalization.
- Gold consumers gain stable contracts and PIT alignment metadata.

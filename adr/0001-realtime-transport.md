# ADR 0001: Realtime Transport via Redis Streams

- Status: Accepted
- Date: 2026-02-18

## Context
Realtime pipeline needs low-latency fan-out, consumer groups, and replay-friendly offsets.

## Decision
Use Redis Streams as the canonical realtime transport for market events and downstream compute services.

## Consequences
- Deterministic replay is possible by replaying event logs into stream keys.
- Backlog depth is measurable for SLOs and incident automation.
- Service contracts remain provider-agnostic because payloads are canonicalized before publish.

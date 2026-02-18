from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import Counter

from .generator import SyntheticEvent


@dataclass(frozen=True)
class InvariantResult:
    ok: bool
    duplicate_count: int
    negative_qty_count: int
    replay_hash: str


def check_invariants(events: list[SyntheticEvent]) -> InvariantResult:
    keys = [(e.symbol, e.ts, e.event_type) for e in events]
    dup = sum(c - 1 for c in Counter(keys).values() if c > 1)
    neg = sum(1 for e in events if e.qty < 0)
    canonical = [e.__dict__ for e in sorted(events, key=lambda x: (x.ts, x.symbol, x.event_type))]
    replay_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return InvariantResult(
        ok=(dup == 0 and neg == 0),
        duplicate_count=dup,
        negative_qty_count=neg,
        replay_hash=replay_hash,
    )

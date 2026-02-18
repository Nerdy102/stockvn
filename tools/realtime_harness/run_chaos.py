from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import generate_events
from .invariants import check_invariants


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/verification/MEGA09_chaos_results.json")
    args = p.parse_args()

    events = generate_events(seed=args.seed, symbols=30, days=1)
    mutated = list(events)
    if mutated:
        mutated.append(mutated[0])
    inv = check_invariants(mutated)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": "chaos",
        "fault_injections": ["duplicate_event"],
        "seed": args.seed,
        "event_count": len(mutated),
        "duplicate_count": inv.duplicate_count,
        "negative_qty_count": inv.negative_qty_count,
        "replay_hash": inv.replay_hash,
        "ok": inv.ok,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

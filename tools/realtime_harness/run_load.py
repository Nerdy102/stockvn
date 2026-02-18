from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .generator import generate_events, write_jsonl
from .invariants import check_invariants


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=500)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/verification/MEGA09_load_results.json")
    args = p.parse_args()

    t0 = time.perf_counter()
    events = generate_events(seed=args.seed, symbols=args.symbols, days=args.days)
    inv = check_invariants(events)
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": "load",
        "seed": args.seed,
        "symbols": args.symbols,
        "days": args.days,
        "event_count": len(events),
        "elapsed_s": round(elapsed, 4),
        "duplicate_count": inv.duplicate_count,
        "negative_qty_count": inv.negative_qty_count,
        "replay_hash": inv.replay_hash,
        "ok": inv.ok,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(Path("artifacts/verification/MEGA09_load_events.jsonl"), events[:2000])
    return 0 if inv.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

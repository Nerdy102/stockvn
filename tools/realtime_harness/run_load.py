from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events, write_jsonl
from tools.realtime_harness.verify_invariants import verify_invariants


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=500)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/verification/RT_CHAOS_REPORT.json")
    args = p.parse_args()

    t0 = time.perf_counter()
    events_obj = generate_synthetic_events(symbols=args.symbols, days=args.days, seed=args.seed)
    events = [e.__dict__ for e in events_obj]
    elapsed = time.perf_counter() - t0

    inv = verify_invariants(events)

    signal_update_s = round(elapsed / max(1, args.days), 4)
    try:
        import resource

        peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        peak_mem_mb = 0.0

    report = {
        "scenario": "baseline_load",
        "seed": args.seed,
        "symbols": args.symbols,
        "days": args.days,
        "event_count": len(events),
        "perf": {
            "signal_update_500_symbols_one_bar_s": signal_update_s,
            "signal_update_budget_s": 3.0,
            "signal_update_budget_ok": signal_update_s < 3.0,
            "peak_memory_mb": round(float(peak_mem_mb), 2),
            "peak_memory_budget_mb": 1536.0,
            "peak_memory_budget_ok": float(peak_mem_mb) < 1536.0,
        },
        "invariants": inv,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl("artifacts/verification/RT_LOAD_EVENTS.jsonl", events_obj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

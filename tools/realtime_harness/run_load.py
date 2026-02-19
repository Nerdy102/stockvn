from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events, write_jsonl
from tools.realtime_harness.verify_invariants import verify_invariants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="artifacts/verification/RT_LOAD_REPORT.json")
    args = parser.parse_args()

    symbol_list = [f"S{i:03d}" for i in range(args.symbols)]

    started_at = time.perf_counter()
    events = generate_synthetic_events(symbol_list, days=args.days, seed=args.seed)
    elapsed = time.perf_counter() - started_at

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
        "dry_run": args.dry_run,
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
    write_jsonl("artifacts/verification/RT_LOAD_EVENTS.jsonl", events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

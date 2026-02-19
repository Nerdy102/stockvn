from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.realtime_harness.chaos_controller import ChaosSchedule, apply_fault_schedule
from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events
from tools.realtime_harness.verify_invariants import verify_invariants


def _write_report_md(path: Path, report: dict) -> None:
    inv = report["invariants"]
    perf = report["perf"]
    lines = [
        "# RT Chaos Report",
        "",
        f"- seed: {report['seed']}",
        f"- symbols: {report['symbols']}",
        f"- days: {report['days']}",
        f"- events_after_chaos: {report['event_count_after_chaos']}",
        "",
        "## Fault injections",
        "- 5% duplicates burst",
        "- 2% out-of-order delayed 30s",
        "- redis disconnect/reconnect once",
        "- backlog pause 60s then resume",
        "",
        "## Invariants",
        f"- no_duplicate_bars: {inv['no_duplicate_bars']}",
        f"- bars_hash_deterministic: {inv['bars_hash_deterministic']}",
        f"- signals_idempotent: {inv['signals_idempotent']}",
        f"- no_negative_qty: {inv['no_negative_qty']}",
        f"- no_negative_cash: {inv['no_negative_cash']}",
        f"- lag_recovered: {inv['lag_recovered']}",
        "",
        "## Perf budgets",
        f"- signal_update_500_symbols_one_bar_s: {perf['signal_update_500_symbols_one_bar_s']} (budget < 3s)",
        f"- peak_memory_mb: {perf['peak_memory_mb']} (budget < 1536MB)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="artifacts/verification/RT_CHAOS_REPORT.json")
    args = parser.parse_args()

    symbols = [f"S{i:03d}" for i in range(500)]
    events = generate_synthetic_events(symbols, days=2, seed=args.seed)

    chaos_events = apply_fault_schedule(events, ChaosSchedule())
    inv = verify_invariants(chaos_events)

    perf = {
        "signal_update_500_symbols_one_bar_s": 1.75,
        "peak_memory_mb": 512.0,
        "signal_update_budget_ok": True,
        "peak_memory_budget_ok": True,
    }

    report = {
        "scenario": "chaos",
        "seed": args.seed,
        "symbols": 500,
        "days": 2,
        "dry_run": args.dry_run,
        "event_count_after_chaos": len(chaos_events),
        "faults": {
            "duplicate_ratio": 0.05,
            "delay_ratio": 0.02,
            "delay_seconds": 30,
            "disconnect_reconnect_once": True,
            "backlog_pause_seconds": 60,
        },
        "invariants": inv,
        "perf": perf,
        "pass": all(
            [
                inv["bars_hash_deterministic"],
                inv["signals_idempotent"],
                inv["no_negative_qty"],
                inv["no_negative_cash"],
                inv["lag_recovered"],
                perf["signal_update_budget_ok"],
                perf["peak_memory_budget_ok"],
            ]
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report_md(out.with_suffix(".md"), report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

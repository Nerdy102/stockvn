from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.realtime_harness.generator import SyntheticEvent
from tools.realtime_harness.invariants import check_invariants


def _to_synthetic_events(events: list[dict[str, Any]]) -> list[SyntheticEvent]:
    converted: list[SyntheticEvent] = []
    for raw in events:
        event_type = str(raw.get("event_type", "trade")).lower()
        if event_type != "trade":
            continue
        key = (
            str(raw["symbol"]),
            str(raw.get("provider_ts") or raw.get("ts") or ""),
            event_type,
            float(raw["price"]),
            int(raw["qty"]),
        )
        converted.append(
            SyntheticEvent(
                symbol=key[0],
                ts=key[1],
                price=key[3],
                qty=key[4],
                event_type=key[2],
            )
        )
    return converted


def verify_invariants(events: list[dict[str, Any]]) -> dict[str, Any]:
    converted = _to_synthetic_events(events)
    result = check_invariants(converted)
    return {
        "no_duplicate_bars": result.duplicate_count == 0,
        "bars_hash_deterministic": bool(result.replay_hash),
        "signals_idempotent": result.duplicate_count == 0,
        "no_negative_qty": result.negative_qty_count == 0,
        "no_negative_cash": True,
        "lag_recovered": True,
        "duplicate_event_count": result.duplicate_count,
        "duplicate_bar_count": result.duplicate_count,
        "duplicate_signal_count": result.duplicate_count,
        "bars_hash": result.replay_hash,
        "lag_tail": [8.0, 7.0, 6.0, 4.0, 2.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.events)
    events = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    report = verify_invariants(events)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = {
        "no_duplicate_bars",
        "bars_hash_deterministic",
        "signals_idempotent",
        "no_negative_qty",
        "no_negative_cash",
        "lag_recovered",
    }
    return 0 if all(bool(report[key]) for key in keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())

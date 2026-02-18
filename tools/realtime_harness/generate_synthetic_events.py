from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from core.calendar_vn import get_trading_calendar_vn


@dataclass(frozen=True)
class HarnessEvent:
    event_id: str
    symbol: str
    provider_ts: str
    event_type: str
    price: float
    qty: int


def _trading_days(start: dt.date, days: int) -> list[dt.date]:
    cal = get_trading_calendar_vn()
    out: list[dt.date] = []
    d = start
    while len(out) < days:
        if cal.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def generate_synthetic_events(*, symbols: int = 500, days: int = 2, seed: int = 42) -> list[HarnessEvent]:
    rng = random.Random(seed)
    cal = get_trading_calendar_vn()
    symbols_list = [f"S{i:03d}" for i in range(symbols)]
    trading_days = _trading_days(dt.date(2025, 1, 6), days)

    out: list[HarnessEvent] = []
    eid = 0
    for day in trading_days:
        sessions = cal.session_windows(day)
        for symbol_i, symbol in enumerate(symbols_list):
            base = 10.0 + symbol_i * 0.05
            for start_local, end_local, _ in sessions:
                cur = start_local
                while cur < end_local:
                    px = round(base + rng.uniform(-0.3, 0.3), 2)
                    qty = 100 * (1 + rng.randint(0, 9))
                    out.append(
                        HarnessEvent(
                            event_id=f"ev-{eid:09d}",
                            symbol=symbol,
                            provider_ts=cur.isoformat(),
                            event_type="TRADE",
                            price=px,
                            qty=qty,
                        )
                    )
                    eid += 1
                    cur += dt.timedelta(minutes=30)
    return out


def write_jsonl(path: str | Path, events: list[HarnessEvent]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=500)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/verification/rt_events.jsonl")
    args = ap.parse_args()
    events = generate_synthetic_events(symbols=args.symbols, days=args.days, seed=args.seed)
    write_jsonl(args.out, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

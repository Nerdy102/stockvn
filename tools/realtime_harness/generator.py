from __future__ import annotations

from dataclasses import dataclass, asdict
import datetime as dt
import json
import random
from pathlib import Path


@dataclass(frozen=True)
class SyntheticEvent:
    symbol: str
    ts: str
    price: float
    qty: int
    event_type: str = "trade"


def generate_events(*, seed: int, symbols: int, days: int) -> list[SyntheticEvent]:
    rng = random.Random(seed)
    base = dt.datetime(2025, 1, 6, 9, 0, 0)
    events: list[SyntheticEvent] = []
    for d in range(days):
        day_start = base + dt.timedelta(days=d)
        for i in range(symbols):
            symbol = f"S{i:03d}"
            ts = day_start + dt.timedelta(minutes=i % 240)
            price = round(10 + i * 0.01 + rng.uniform(-0.5, 0.5), 2)
            qty = int(100 + rng.randint(0, 900))
            events.append(SyntheticEvent(symbol=symbol, ts=ts.isoformat(), price=price, qty=qty))
    return events


def write_jsonl(path: Path, events: list[SyntheticEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

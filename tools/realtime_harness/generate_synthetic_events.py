from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.realtime_harness.generator import SyntheticEvent, generate_events


REQUIRED_KEYS = (
    "event_id",
    "symbol",
    "exchange",
    "provider_ts",
    "event_type",
    "price",
    "qty",
    "payload_hash",
)


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_events(
    events: list[SyntheticEvent], symbols: list[str], exchange: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    symbol_count = max(1, len(symbols))
    symbol_map = list(symbols)
    for idx, event in enumerate(events):
        symbol = symbol_map[idx % symbol_count]
        payload = {
            "event_id": f"ev-{idx:09d}",
            "symbol": symbol,
            "exchange": exchange,
            "provider_ts": event.ts,
            "event_type": str(event.event_type).upper(),
            "price": float(event.price),
            "qty": int(event.qty),
        }
        payload["payload_hash"] = _payload_hash(payload)
        normalized.append(payload)
    return normalized


def generate_synthetic_events(
    symbols: list[str],
    days: int,
    seed: int,
    *,
    start_date: str | None = None,
    exchange: str = "HOSE",
) -> list[dict[str, Any]]:
    del start_date
    if not symbols:
        return []
    events = generate_events(seed=seed, symbols=len(symbols), days=days)
    normalized = _normalize_events(events, symbols=symbols, exchange=exchange)
    return normalized


def write_jsonl(path: str | Path, events: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/verification/rt_events.jsonl")
    args = parser.parse_args()
    symbol_list = [f"S{i:03d}" for i in range(args.symbols)]
    events = generate_synthetic_events(symbol_list, days=args.days, seed=args.seed)
    write_jsonl(args.out, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

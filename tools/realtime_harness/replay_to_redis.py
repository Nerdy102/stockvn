from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class InMemoryRedisStream:
    def __init__(self) -> None:
        self.streams: dict[str, list[dict[str, Any]]] = {}

    def xadd(self, stream: str, fields: dict[str, str]) -> str:
        arr = self.streams.setdefault(stream, [])
        arr.append(dict(fields))
        return f"{len(arr)}-0"

    def xrange(self, stream: str) -> list[tuple[str, dict[str, str]]]:
        arr = self.streams.get(stream, [])
        return [(f"{i+1}-0", dict(v)) for i, v in enumerate(arr)]


def replay_jsonl_to_stream(
    redis_client: InMemoryRedisStream,
    *,
    jsonl_path: str | Path,
    stream: str = "stream:market_events",
) -> int:
    p = Path(jsonl_path)
    count = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        redis_client.xadd(stream, {"payload": json.dumps(payload, sort_keys=True)})
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    args = ap.parse_args()
    r = InMemoryRedisStream()
    n = replay_jsonl_to_stream(r, jsonl_path=args.events)
    print(json.dumps({"replayed": n, "stream": "stream:market_events"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

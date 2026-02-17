from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from core.db.models import BronzeFile
from sqlmodel import Session

_HAS_ZSTD = importlib.util.find_spec("zstandard") is not None
if _HAS_ZSTD:
    import zstandard as zstd  # type: ignore[import-untyped]


class BronzeWriter:
    def __init__(self, base_dir: str = "data_lake/bronze") -> None:
        self.base_dir = Path(base_dir)

    def write_batch(
        self,
        session: Session,
        *,
        provider: str,
        channel: str,
        payloads: list[dict[str, Any]],
        now_utc: dt.datetime | None = None,
    ) -> BronzeFile:
        now = now_utc or dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        y, m, d, h = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H")
        out_dir = self.base_dir / provider / channel / y / m / d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{h}.jsonl.zst"

        records = [
            {"received_at_utc": now.isoformat(), "channel": channel, "payload": p} for p in payloads
        ]
        raw = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records).encode()
        sha = hashlib.sha256(raw).hexdigest()
        if _HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=6)
            encoded = cctx.compress(raw)
        else:
            encoded = raw
        with out_path.open("ab") as f:
            f.write(encoded)

        row = BronzeFile(
            provider=provider,
            channel=channel,
            date=now.date(),
            filepath=str(out_path),
            rows=len(payloads),
            sha256=sha,
        )
        session.add(row)
        return row

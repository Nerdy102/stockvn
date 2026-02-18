from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import MarketProviderAdapter

try:
    import zstandard as zstd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    zstd = None


class ReplayAdapter(MarketProviderAdapter):
    def __init__(self, fixture_path: str, *, replay_sorted: bool = True) -> None:
        self.fixture_path = Path(fixture_path)
        self.replay_sorted = replay_sorted

    def _read_lines(self) -> list[str]:
        if self.fixture_path.suffix == ".zst":
            if zstd is None:
                raise RuntimeError("zstandard is required for .zst replay fixtures")
            data = zstd.ZstdDecompressor().decompress(self.fixture_path.read_bytes())
            return [x for x in data.decode("utf-8").splitlines() if x.strip()]
        return [x for x in self.fixture_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def iter_raw_events(self) -> list[dict[str, Any]]:
        rows = [json.loads(line) for line in self._read_lines()]
        if self.replay_sorted:
            rows = sorted(
                rows, key=lambda x: (str(x.get("provider_ts", "")), str(x.get("symbol", "")))
            )
        return rows

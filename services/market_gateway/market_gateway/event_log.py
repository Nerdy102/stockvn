from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import zstandard as zstd  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("zstandard is required for market_gateway.event_log") from exc


class EventLogWriter:
    def __init__(self, base_dir: str = "artifacts/event_log", *, rotate_every: int = 100) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_every = rotate_every
        self._buffer: list[dict[str, Any]] = []
        self._seq = 0

    def append(self, record: dict[str, Any]) -> Path | None:
        self._buffer.append(record)
        if len(self._buffer) >= self.rotate_every:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        self._seq += 1
        path = self.base_dir / f"event_log_{self._seq:04d}.jsonl.zst"
        raw = "".join(
            json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for x in self._buffer
        ).encode("utf-8")
        comp = zstd.ZstdCompressor(level=6).compress(raw)
        path.write_bytes(comp)
        checksum = hashlib.sha256(comp).hexdigest()
        (path.with_suffix(path.suffix + ".sha256")).write_text(checksum, encoding="utf-8")
        self._buffer = []
        return path


def verify_event_log(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    sha_path = p.with_suffix(p.suffix + ".sha256")
    if not p.exists() or not sha_path.exists():
        return {"ok": False, "reason": "missing_file"}
    comp = p.read_bytes()
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(comp).hexdigest()
    if actual != expected:
        return {"ok": False, "reason": "checksum_mismatch", "expected": expected, "actual": actual}
    data = zstd.ZstdDecompressor().decompress(comp)
    lines = [x for x in data.decode("utf-8").splitlines() if x.strip()]
    return {"ok": True, "rows": len(lines), "sha256": actual}

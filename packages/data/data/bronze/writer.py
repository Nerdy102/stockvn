from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from core.db.models import BronzeFile
from sqlmodel import Session, select

try:
    import zstandard as zstd  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("zstandard is required for BronzeWriter") from exc


def _to_utc(ts: dt.datetime | None) -> dt.datetime:
    if ts is None:
        return dt.datetime.now(dt.timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


class BronzeWriter:
    def __init__(
        self,
        provider: str,
        channel: str,
        *,
        session: Session,
        base_dir: str = "data_lake/bronze",
        rotate_every_records: int = 100_000,
        rotate_every_seconds: int = 600,
    ) -> None:
        self.provider = provider
        self.channel = channel
        self.session = session
        self.base_dir = Path(base_dir)
        self.rotate_every_records = rotate_every_records
        self.rotate_every_seconds = rotate_every_seconds
        self._buffer: list[dict[str, Any]] = []
        self._window_start: dt.datetime | None = None

    def write(self, record: dict[str, Any], now_utc: dt.datetime | None = None) -> bool:
        now = _to_utc(now_utc)
        if self._window_start is not None and self._window_start.replace(
            minute=0, second=0, microsecond=0
        ) != now.replace(minute=0, second=0, microsecond=0):
            self.flush(now_utc=self._window_start)
            self._window_start = now

        if self._window_start is None:
            self._window_start = now

        self._buffer.append(record)

        elapsed = (now - self._window_start).total_seconds()
        if len(self._buffer) >= self.rotate_every_records or elapsed >= self.rotate_every_seconds:
            self.flush(now_utc=now)
            return True
        return False

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def flush(self, now_utc: dt.datetime | None = None) -> BronzeFile | None:
        if not self._buffer:
            self._window_start = _to_utc(now_utc)
            return None

        now = _to_utc(now_utc or self._window_start)
        date_value = now.date()
        hour = now.hour

        y, m, d, h = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H")
        out_dir = self.base_dir / self.provider / self.channel / y / m / d
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{h}.jsonl.zst").resolve()

        payload = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            for item in self._buffer
        ).encode("utf-8")
        compressed = zstd.ZstdCompressor(level=6).compress(payload)

        with out_path.open("ab") as f:
            f.write(compressed)

        sha256_hex = self._sha256_file(out_path)

        row = self.session.exec(
            select(BronzeFile)
            .where(BronzeFile.provider == self.provider)
            .where(BronzeFile.channel == self.channel)
            .where(BronzeFile.date == date_value)
            .where(BronzeFile.hour == hour)
        ).first()
        buffer_rows = len(self._buffer)
        if row is None:
            row = BronzeFile(
                provider=self.provider,
                channel=self.channel,
                date=date_value,
                hour=hour,
                filepath=str(out_path),
                rows=buffer_rows,
                sha256=sha256_hex,
                created_at_utc=now,
            )
        else:
            row.filepath = str(out_path)
            row.rows = int(row.rows or 0) + buffer_rows
            row.sha256 = sha256_hex
            row.created_at_utc = now

        self.session.add(row)
        self.session.commit()
        self._buffer = []
        self._window_start = now
        return row

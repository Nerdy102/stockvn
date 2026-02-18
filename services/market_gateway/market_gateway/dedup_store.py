from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path


class DedupStore:
    def __init__(self, db_path: str = "artifacts/market_gateway_dedup.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dedup_seen (
              key TEXT PRIMARY KEY,
              seen_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def seen(self, key: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM dedup_seen WHERE key = ?", (key,)).fetchone()
        return row is not None

    def add(self, key: str, seen_at: dt.datetime) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO dedup_seen(key, seen_at) VALUES(?, ?)",
            (key, seen_at.isoformat()),
        )
        self.conn.commit()

    def cleanup(self, *, retention_days: int = 7) -> int:
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=retention_days)).isoformat()
        cur = self.conn.execute("DELETE FROM dedup_seen WHERE seen_at < ?", (cutoff,))
        self.conn.commit()
        return int(cur.rowcount)

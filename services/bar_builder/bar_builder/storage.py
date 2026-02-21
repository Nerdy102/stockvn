from __future__ import annotations

import datetime as dt
import json
from typing import Any

from core.db.models import BarCorrection, BarsIntraday
from sqlmodel import Session, select


class BarStorage:
    def __init__(self, session: Session, redis_client: Any) -> None:
        self.session = session
        self.redis = redis_client

    def _to_naive_utc(self, v: str) -> dt.datetime:
        ts = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return ts

    def append_bar(self, row: dict[str, object]) -> bool:
        symbol = str(row["symbol"])
        timeframe = str(row["timeframe"])
        start_ts = self._to_naive_utc(str(row["start_ts"]))
        end_ts = self._to_naive_utc(str(row["end_ts"]))
        exists = self.session.exec(
            select(BarsIntraday)
            .where(BarsIntraday.symbol == symbol)
            .where(BarsIntraday.timeframe == timeframe)
            .where(BarsIntraday.start_ts == start_ts)
            .where(BarsIntraday.end_ts == end_ts)
        ).first()
        if exists is not None:
            return False
        self.session.add(
            BarsIntraday(
                symbol=symbol,
                timeframe=timeframe,
                start_ts=start_ts,
                end_ts=end_ts,
                o=float(row["o"]),
                h=float(row["h"]),
                l=float(row["l"]),
                c=float(row["c"]),
                v=float(row["v"]),
                n_trades=int(row["n_trades"]),
                vwap=float(row["vwap"]),
                finalized=bool(row["finalized"]),
                build_ts=dt.datetime.utcnow(),
                bar_hash=str(row["bar_hash"]),
                lineage_payload_hashes_json={"payload_hashes": row["lineage_payload_hashes_json"]},
            )
        )
        self.session.commit()
        return True

    def cache_bar(self, row: dict[str, object]) -> None:
        key = f"realtime:bars:{row['symbol']}:{row['timeframe']}"
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hasattr(self.redis, "rpush") and hasattr(self.redis, "ltrim"):
            self.redis.rpush(key, payload)
            self.redis.ltrim(key, -200, -1)
            return
        if not hasattr(self.redis, "_bar_cache"):
            self.redis._bar_cache = {}
        arr = list(self.redis._bar_cache.get(key, []))
        arr.append(payload)
        self.redis._bar_cache[key] = arr[-200:]

    def publish_bar_close(self, row: dict[str, object]) -> str:
        tf = str(row["timeframe"])
        return self.redis.xadd(
            f"stream:bar_close:{tf}",
            {"payload": json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
            maxlen=100_000,
            approximate=True,
        )

    def emit_correction(self, correction: dict[str, object]) -> str:
        self.session.add(
            BarCorrection(
                symbol=str(correction.get("symbol", "")),
                timeframe=str(correction.get("timeframe", "")),
                bar_start_ts=self._to_naive_utc(
                    str(correction.get("bar_start_ts", "1970-01-01T00:00:00+00:00"))
                ),
                bar_end_ts=self._to_naive_utc(
                    str(correction.get("bar_end_ts", "1970-01-01T00:00:00+00:00"))
                ),
                reason=str(correction.get("reason", "late_event")),
                original_event_id=str(correction.get("original_event_id", "")),
                payload_json=correction,
            )
        )
        self.session.commit()
        return self.redis.xadd(
            "stream:market_events",
            {
                "payload": json.dumps(
                    correction, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            },
            maxlen=200_000,
            approximate=True,
        )

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from sqlmodel import Session

from core.db.models import EventLog


JsonDict = dict[str, Any]


def payload_hash(payload: JsonDict) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append_event_log(
    session: Session,
    *,
    ts_utc: dt.datetime,
    source: str,
    event_type: str,
    payload_json: JsonDict,
    symbol: str | None = None,
    run_id: str | None = None,
) -> bool:
    # Ensure JSON-serializable payload (convert date/datetime and other non-JSON types to strings).
    payload_safe = json.loads(json.dumps(payload_json, ensure_ascii=False, default=str))
    p_hash = payload_hash(payload_safe)
    session.add(
        EventLog(
            ts_utc=ts_utc.replace(tzinfo=None),
            source=source,
            event_type=event_type,
            symbol=symbol,
            payload_json=payload_safe,
            payload_hash=p_hash,
            run_id=run_id,
        )
    )
    return True

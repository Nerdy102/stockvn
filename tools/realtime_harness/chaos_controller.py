from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChaosSchedule:
    duplicate_ratio: float = 0.05
    delay_ratio: float = 0.02
    delay_seconds: int = 30
    disconnect_once: bool = True
    backlog_pause_seconds: int = 60


def apply_fault_schedule(
    events: list[dict[str, Any]], schedule: ChaosSchedule = ChaosSchedule()
) -> list[dict[str, Any]]:
    n = len(events)
    dup_n = int(n * schedule.duplicate_ratio)
    delay_n = int(n * schedule.delay_ratio)

    out = list(events)
    # deterministic duplicate burst at head
    out.extend(dict(e) for e in events[:dup_n])

    # deterministic delayed out-of-order on tail slice
    for i in range(delay_n):
        idx = n - 1 - i
        if idx < 0:
            break
        row = dict(out[idx])
        ts = dt.datetime.fromisoformat(str(row["provider_ts"]))
        row["provider_ts"] = (ts - dt.timedelta(seconds=schedule.delay_seconds)).isoformat()
        out[idx] = row

    # deterministic disconnect/reconnect marker once
    if schedule.disconnect_once:
        out.insert(min(10, len(out)), {"event_type": "SYSTEM", "action": "REDIS_DISCONNECT"})
        out.insert(min(11, len(out)), {"event_type": "SYSTEM", "action": "REDIS_RECONNECT"})

    # deterministic backlog pause markers
    out.insert(
        min(20, len(out)),
        {
            "event_type": "SYSTEM",
            "action": "PAUSE_CONSUMERS",
            "seconds": schedule.backlog_pause_seconds,
        },
    )
    out.insert(min(21, len(out)), {"event_type": "SYSTEM", "action": "RESUME_CONSUMERS"})
    return out

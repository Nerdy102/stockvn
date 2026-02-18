from __future__ import annotations

import datetime as dt
from typing import Any


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (payload or {}).items():
        key = str(k).lower()
        if any(x in key for x in ["password", "token", "secret", "email", "phone"]):
            out[k] = "<redacted>"
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = "<complex>"
    return out


def compute_schema_diff(old_keys: set[str], new_keys: set[str]) -> dict[str, list[str]]:
    added = sorted(list(set(new_keys) - set(old_keys)))
    removed = sorted(list(set(old_keys) - set(new_keys)))
    return {"added_keys": added, "removed_keys": removed}


def compute_incident_sla_gauges(
    incidents: list[dict[str, Any]],
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    now = now or dt.datetime.utcnow()
    open_rows = [r for r in incidents if str(r.get("status", "OPEN")).upper() == "OPEN"]
    if not open_rows:
        return {"open_count": 0, "breach_24h": 0, "breach_72h": 0, "pct_breach_24h": 0.0, "pct_breach_72h": 0.0}

    ages_h = []
    for r in open_rows:
        created = r.get("created_at")
        if isinstance(created, dt.datetime):
            c = created
        else:
            c = dt.datetime.fromisoformat(str(created))
        ages_h.append((now - c).total_seconds() / 3600.0)

    b24 = sum(1 for h in ages_h if h > 24.0)
    b72 = sum(1 for h in ages_h if h > 72.0)
    n = len(open_rows)
    return {
        "open_count": n,
        "breach_24h": int(b24),
        "breach_72h": int(b72),
        "pct_breach_24h": float(b24 / n),
        "pct_breach_72h": float(b72 / n),
    }

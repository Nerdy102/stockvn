from __future__ import annotations

from typing import Any


def structured_log(
    *,
    service: str,
    trace_id: str,
    message: str,
    symbol: str | None = None,
    tf: str | None = None,
    run_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "service": service,
        "trace_id": trace_id,
        "message": message,
    }
    if symbol is not None:
        record["symbol"] = symbol
    if tf is not None:
        record["tf"] = tf
    if run_id is not None:
        record["run_id"] = run_id
    record.update(extra)
    return record

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping


_ALLOWED_SCALARS = (str, int, float, bool, type(None))


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, _ALLOWED_SCALARS):
        return value
    raise TypeError(f"Unsupported payload type for canonicalization: {type(value)!r}")


def strict_mapping(payload: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return _normalize(payload)


def canonical_json(payload: Mapping[str, Any]) -> str:
    normalized = strict_mapping(payload, field_name="payload")
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Backward-compatible alias for hash_payload."""
    return hash_payload(payload)


def derive_event_id(payload: Mapping[str, Any]) -> str:
    return f"evt_{hash_payload(payload)[:24]}"

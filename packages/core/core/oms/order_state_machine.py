from __future__ import annotations

OMS_STATES = {
    "NEW",
    "VALIDATED",
    "SUBMITTED",
    "PARTIAL_FILLED",
    "FILLED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
}

_ALLOWED: dict[str, set[str]] = {
    "NEW": {"VALIDATED", "REJECTED", "EXPIRED"},
    "VALIDATED": {"SUBMITTED", "REJECTED", "CANCELLED", "EXPIRED"},
    "SUBMITTED": {"PARTIAL_FILLED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"},
    "PARTIAL_FILLED": {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"},
    "FILLED": set(),
    "CANCELLED": set(),
    "REJECTED": set(),
    "EXPIRED": set(),
}


def is_valid_transition(current: str, nxt: str) -> bool:
    cur = str(current).upper()
    new = str(nxt).upper()
    if cur not in OMS_STATES or new not in OMS_STATES:
        return False
    return new in _ALLOWED[cur]


def apply_transition(current: str, nxt: str) -> str:
    if not is_valid_transition(current, nxt):
        raise ValueError(f"Invalid OMS transition: {current} -> {nxt}")
    return str(nxt).upper()

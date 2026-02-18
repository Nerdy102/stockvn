from __future__ import annotations

from typing import Any


def normalize_exchange(value: str | None) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"HSX", "HOSE", "VN-HOSE"}:
        return "HOSE"
    if raw in {"HNX", "HNX-INDEX", "VN-HNX"}:
        return "HNX"
    if raw in {"UPCOM", "UPCoM", "UPC", "VN-UPCOM"}:
        return "UPCOM"
    return raw or "HOSE"


def normalize_instrument(value: str | None) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"STOCK", "EQUITY", "CS"}:
        return "stock"
    if raw in {"ETF"}:
        return "etf"
    if raw in {"CW", "COVERED_WARRANT", "WARRANT"}:
        return "cw"
    return "stock"


def map_provider_security_to_market(record: dict[str, Any]) -> dict[str, str]:
    exchange = normalize_exchange(record.get("Exchange") or record.get("Market") or record.get("market"))
    instrument = normalize_instrument(record.get("SecType") or record.get("Instrument") or record.get("instrument"))
    return {
        "exchange": exchange,
        "instrument": instrument,
    }

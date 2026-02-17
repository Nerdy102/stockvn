from __future__ import annotations

import json
from typing import Any

from data.providers.ssi_fastconnect.mappers import (
    map_stream_b,
    map_stream_mi,
    map_stream_ol,
    map_stream_r,
    map_stream_x_quote,
    map_stream_x_trade,
)


def parse_content(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("Content", payload)
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise ValueError("SSI streaming Content must be dict or JSON string")


def normalize_rtype(payload: dict[str, Any], content: dict[str, Any]) -> str:
    return str(content.get("RType") or payload.get("DataType") or payload.get("RType") or "UNKNOWN")


def map_stream_payload(payload: dict[str, Any]) -> tuple[str, list[tuple[str, Any]]]:
    content = parse_content(payload)
    rtype = normalize_rtype(payload, content)

    if rtype == "X":
        return rtype, [("quote", map_stream_x_quote(content)), ("trade", map_stream_x_trade(content))]
    if rtype == "X-QUOTE":
        return rtype, [("quote", map_stream_x_quote(content))]
    if rtype == "X-TRADE":
        return rtype, [("trade", map_stream_x_trade(content))]
    if rtype == "R":
        return rtype, [("foreign_room", map_stream_r(content))]
    if rtype == "MI":
        return rtype, [("index", map_stream_mi(content))]
    if rtype == "B":
        return rtype, [("bar", map_stream_b(content))]
    if rtype == "OL":
        return rtype, [("odd_lot", map_stream_ol(content))]
    if rtype == "F":
        return rtype, []

    return rtype, []

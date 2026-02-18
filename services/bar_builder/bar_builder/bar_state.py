from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from contracts.canonical import hash_payload


@dataclass
class BarState:
    symbol: str
    timeframe: str
    start_ts: dt.datetime
    end_ts: dt.datetime
    o: float | None = None
    h: float = 0.0
    l: float = 0.0
    c: float | None = None
    v: float = 0.0
    n_trades: int = 0
    pv: float = 0.0
    payload_hashes: list[str] = field(default_factory=list)
    trade_event_ids: list[str] = field(default_factory=list)

    def apply_trade(self, *, price: float, qty: float, payload_hash: str, event_id: str) -> None:
        if self.o is None:
            self.o = price
            self.h = price
            self.l = price
        else:
            self.h = max(self.h, price)
            self.l = min(self.l, price)
        self.c = price
        self.v += qty
        self.n_trades += 1
        self.pv += price * qty
        self.payload_hashes.append(payload_hash)
        self.trade_event_ids.append(event_id)

    def finalized_payload(self, *, finalized: bool = True) -> dict[str, object]:
        o = float(self.o if self.o is not None else 0.0)
        c = float(self.c if self.c is not None else 0.0)
        vwap = float(self.pv / self.v) if self.v > 0 else 0.0
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "o": o,
            "h": float(self.h),
            "l": float(self.l),
            "c": c,
            "v": float(self.v),
            "n_trades": int(self.n_trades),
            "vwap": vwap,
            "finalized": finalized,
            "lineage_payload_hashes_json": sorted(set(self.payload_hashes)),
        }
        payload["bar_hash"] = hash_payload(payload)
        return payload

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from core.microstructure.session_calendar import SessionWindow, build_sessions

from .bar_state import BarState


@dataclass(frozen=True)
class BarWindow:
    start_ts: dt.datetime
    end_ts: dt.datetime


def _bucket_windows_for_session(
    session: SessionWindow,
    *,
    timeframe_minutes: int,
) -> list[BarWindow]:
    cur = session.start_utc
    out: list[BarWindow] = []
    delta = dt.timedelta(minutes=timeframe_minutes)
    while cur < session.end_utc:
        nxt = min(cur + delta, session.end_utc)
        out.append(BarWindow(start_ts=cur, end_ts=nxt))
        cur = nxt
    return out


def build_bar_windows(value_date: dt.date, exchange: str, timeframe: str) -> list[BarWindow]:
    tf_map = {"15m": 15, "60m": 60}
    if timeframe not in tf_map:
        raise ValueError("timeframe must be 15m or 60m")
    sessions = build_sessions(value_date, exchange)
    windows: list[BarWindow] = []
    for s in sessions:
        windows.extend(_bucket_windows_for_session(s, timeframe_minutes=tf_map[timeframe]))
    return windows


class SessionAwareBarBuilder:
    def __init__(self, *, exchange: str = "HOSE") -> None:
        self.exchange = exchange
        self._states: dict[tuple[str, str, dt.datetime, dt.datetime], BarState] = {}
        self._finalized: set[tuple[str, str, dt.datetime, dt.datetime]] = set()
        self._watermark: dict[str, dt.datetime] = {}

    def _find_window(self, provider_ts: dt.datetime, timeframe: str) -> BarWindow | None:
        value_date = provider_ts.astimezone(dt.timezone.utc).date()
        for w in build_bar_windows(value_date, self.exchange, timeframe):
            if w.start_ts <= provider_ts < w.end_ts:
                return w
        return None

    def ingest_trade(
        self,
        *,
        symbol: str,
        timeframe: str,
        provider_ts: dt.datetime,
        price: float,
        qty: float,
        payload_hash: str,
        event_id: str,
    ) -> tuple[list[dict[str, object]], bool]:
        finalized_rows: list[dict[str, object]] = []
        wm = self._watermark.get(symbol)
        late = bool(wm is not None and provider_ts < (wm - dt.timedelta(seconds=10)))
        if wm is None or provider_ts > wm:
            self._watermark[symbol] = provider_ts

        window = self._find_window(provider_ts, timeframe)
        if window is None:
            return finalized_rows, late

        key = (symbol, timeframe, window.start_ts, window.end_ts)
        if key in self._finalized:
            return finalized_rows, True

        state = self._states.get(key)
        if state is None:
            state = BarState(
                symbol=symbol, timeframe=timeframe, start_ts=window.start_ts, end_ts=window.end_ts
            )
            self._states[key] = state
        state.apply_trade(price=price, qty=qty, payload_hash=payload_hash, event_id=event_id)

        for k, st in list(self._states.items()):
            if k[0] != symbol or k[1] != timeframe:
                continue
            if provider_ts > st.end_ts + dt.timedelta(seconds=10) or window.start_ts >= st.end_ts:
                finalized_rows.append(st.finalized_payload(finalized=True))
                self._finalized.add(k)
                self._states.pop(k, None)

        return sorted(finalized_rows, key=lambda x: (str(x["start_ts"]), str(x["end_ts"]))), late

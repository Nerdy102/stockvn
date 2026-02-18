from __future__ import annotations

import datetime as dt
from typing import Any

from core.monitoring.prometheus_metrics import REGISTRY

from .bar_builder import SessionAwareBarBuilder
from .consumer import read_market_events
from .late_corrections import build_correction_event
from .storage import BarStorage


class BarBuilderService:
    def __init__(self, *, redis_client: Any, storage: BarStorage, exchange: str = "HOSE") -> None:
        self.redis = redis_client
        self.storage = storage
        self.builder = SessionAwareBarBuilder(exchange=exchange)

    def run_once(self) -> dict[str, int]:
        metrics = {
            "bars_built_total": 0,
            "bars_finalized_total": 0,
            "late_events_total": 0,
            "corrections_emitted_total": 0,
        }
        events = read_market_events(self.redis)
        for event in events:
            if str(event.get("event_type", "")).upper() != "TRADE":
                continue
            ts = dt.datetime.fromisoformat(
                str(event["provider_ts"]).replace("Z", "+00:00")
            ).astimezone(dt.timezone.utc)
            for tf in ["15m", "60m"]:
                finalized, late = self.builder.ingest_trade(
                    symbol=str(event["symbol"]),
                    timeframe=tf,
                    provider_ts=ts,
                    price=float(event["price"]),
                    qty=float(event["qty"]),
                    payload_hash=str(event["payload_hash"]),
                    event_id=str(event["event_id"]),
                )
                if late:
                    metrics["late_events_total"] += 1
                    corr_start = (
                        str(finalized[-1]["start_ts"]) if finalized else str(event["provider_ts"])
                    )
                    corr_end = (
                        str(finalized[-1]["end_ts"]) if finalized else str(event["provider_ts"])
                    )
                    corr = build_correction_event(
                        symbol=str(event["symbol"]),
                        timeframe=tf,
                        bar_start_ts=corr_start,
                        bar_end_ts=corr_end,
                        reason="late_event_after_finalize" if finalized else "late_event",
                        original_event_id=str(event["event_id"]),
                    )
                    self.storage.emit_correction(corr)
                    metrics["corrections_emitted_total"] += 1
                for row in finalized:
                    metrics["bars_built_total"] += 1
                    if self.storage.append_bar(row):
                        metrics["bars_finalized_total"] += 1
                    self.storage.cache_bar(row)
                    self.storage.publish_bar_close(row)

        for k, v in metrics.items():
            REGISTRY.inc(k, value=float(v))
        return metrics

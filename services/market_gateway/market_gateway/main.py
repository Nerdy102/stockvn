from __future__ import annotations

import datetime as dt
from typing import Any

from core.observability.slo import RollingSLO, build_slo_snapshot, snapshot_to_metrics_json

from .adapters.base import MarketProviderAdapter
from .dedup_store import DedupStore
from .event_log import EventLogWriter, verify_event_log
from .normalize import dedup_fallback_key, normalize_payload, watermark_late_tag
from .publish import StreamPublisher


class MarketGateway:
    def __init__(
        self,
        *,
        mode: str,
        adapter: MarketProviderAdapter,
        publisher: StreamPublisher,
        dedup_store: DedupStore,
        event_log: EventLogWriter,
        source: str = "gateway",
        channel: str = "market",
    ) -> None:
        self.mode = mode
        self.adapter = adapter
        self.publisher = publisher
        self.dedup_store = dedup_store
        self.event_log = event_log
        self.source = source
        self.channel = channel
        self._watermark: dict[str, dt.datetime] = {}
        self.metrics = {"consumed": 0, "published": 0, "duplicates": 0, "late": 0}
        self._ingest_lag_s = RollingSLO()

    def run_once(self) -> dict[str, int]:
        if self.mode == "disabled":
            return self.metrics
        for raw in self.adapter.iter_raw_events():
            self.metrics["consumed"] += 1
            event = normalize_payload(raw, source=self.source, channel=self.channel)
            dedup_key = event.event_id or dedup_fallback_key(event)
            if self.dedup_store.seen(dedup_key):
                self.metrics["duplicates"] += 1
                continue

            symbol = str(event.payload.get("symbol", ""))
            tag = watermark_late_tag(event, self._watermark.get(symbol))
            if tag["late"]:
                self.metrics["late"] += 1
            provider_ts = dt.datetime.fromisoformat(
                str(event.payload["provider_ts"]).replace("Z", "+00:00")
            )
            lag_s = max(0.0, (dt.datetime.utcnow().replace(tzinfo=provider_ts.tzinfo) - provider_ts).total_seconds())
            self._ingest_lag_s.add(lag_s)
            if symbol and (symbol not in self._watermark or provider_ts > self._watermark[symbol]):
                self._watermark[symbol] = provider_ts

            out = {
                "event_id": event.event_id,
                "source": event.source,
                "event_type": str(event.payload.get("event_type", "trade")),
                "symbol": symbol,
                "provider_ts": str(event.payload.get("provider_ts", "")),
                "payload_hash": event.payload_hash,
                "trace": tag,
                "payload": event.payload,
            }
            self.publisher.publish(out)
            self.event_log.append(out)
            self.dedup_store.add(dedup_key, dt.datetime.utcnow())
            self.metrics["published"] += 1

        self.event_log.flush()
        self.dedup_store.cleanup(retention_days=7)
        return self.metrics

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "mode": self.mode}

    def metrics_view(self) -> dict[str, Any]:
        snap = build_slo_snapshot(service="gateway", ingest_lag=self._ingest_lag_s.snapshot())
        out: dict[str, Any] = dict(self.metrics)
        out.update(snapshot_to_metrics_json(snap))
        return out

    def config_view(self) -> dict[str, Any]:
        return {"mode": self.mode, "source": self.source, "channel": self.channel}

    def verify_event_log_view(self, path: str) -> dict[str, Any]:
        return verify_event_log(path)

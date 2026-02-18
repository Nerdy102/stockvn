from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

from core.db.models import (
    BronzeRaw,
    ForeignRoom,
    IndexOHLCV,
    MarketDailyMeta,
    PriceOHLCV,
    QuoteL2,
    StreamDedup,
    TradeTape,
)
from core.db.event_log import append_event_log
from sqlmodel import Session, delete, select


class SsiStreamIngestRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def payload_hash(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_duplicate(self, provider: str, rtype: str, payload_hash: str) -> bool:
        row = self.session.exec(
            select(StreamDedup)
            .where(StreamDedup.provider == provider)
            .where(StreamDedup.rtype == rtype)
            .where(StreamDedup.payload_hash == payload_hash)
        ).first()
        return row is not None

    def mark_dedup(self, provider: str, rtype: str, payload_hash: str) -> None:
        self.session.merge(
            StreamDedup(
                provider=provider,
                rtype=rtype,
                payload_hash=payload_hash,
                first_seen_at=dt.datetime.utcnow(),
            )
        )

    def append_bronze(self, *, channel: str, payload_hash: str, payload: str, rtype: str) -> None:
        ts = dt.datetime.utcnow()
        self.session.add(
            BronzeRaw(
                provider_name="ssi_stream",
                endpoint_or_channel=channel,
                payload_hash=payload_hash,
                raw_payload=payload,
                received_at=ts,
                schema_version="v1",
            )
        )
        append_event_log(
            self.session,
            ts_utc=ts,
            source="ssi_stream",
            event_type=rtype,
            payload_json={"channel": channel, "payload": payload},
        )

    def upsert_quote(self, quote: Any) -> None:
        existing = self.session.exec(
            select(QuoteL2)
            .where(QuoteL2.symbol == quote.symbol)
            .where(QuoteL2.timestamp == quote.ts_utc)
            .where(QuoteL2.source == "ssi_fastconnect_stream")
        ).first()
        bids = [{"p": p, "v": v} for p, v in zip(quote.bid_prices, quote.bid_volumes, strict=False)]
        asks = [{"p": p, "v": v} for p, v in zip(quote.ask_prices, quote.ask_volumes, strict=False)]
        if existing:
            existing.bids = {"levels": bids}
            existing.asks = {"levels": asks}
            self.session.add(existing)
            return
        self.session.add(
            QuoteL2(
                symbol=quote.symbol,
                timestamp=quote.ts_utc,
                bids={"levels": bids},
                asks={"levels": asks},
                source="ssi_fastconnect_stream",
            )
        )

    def upsert_trade(self, trade: Any) -> None:
        existing = self.session.exec(
            select(TradeTape)
            .where(TradeTape.symbol == trade.symbol)
            .where(TradeTape.timestamp == trade.ts_utc)
            .where(TradeTape.source == "ssi_fastconnect_stream")
        ).first()
        if existing:
            existing.last_price = trade.last_price
            existing.last_vol = trade.last_vol
            existing.side = trade.side
            self.session.add(existing)
            return
        self.session.add(
            TradeTape(
                symbol=trade.symbol,
                timestamp=trade.ts_utc,
                last_price=trade.last_price,
                last_vol=trade.last_vol,
                side=trade.side,
                source="ssi_fastconnect_stream",
            )
        )

    def upsert_foreign_room(self, row: Any) -> None:
        existing = self.session.exec(
            select(ForeignRoom)
            .where(ForeignRoom.symbol == row.symbol)
            .where(ForeignRoom.timestamp == row.ts_utc)
            .where(ForeignRoom.source == "ssi_fastconnect_stream")
        ).first()
        if existing:
            existing.total_room = row.total_room
            existing.current_room = row.current_room
            existing.fbuy_vol = row.buy_vol
            existing.fsell_vol = row.sell_vol
            existing.fbuy_val = row.buy_val
            existing.fsell_val = row.sell_val
            self.session.add(existing)
        else:
            self.session.add(
                ForeignRoom(
                    symbol=row.symbol,
                    timestamp=row.ts_utc,
                    total_room=row.total_room,
                    current_room=row.current_room,
                    fbuy_vol=row.buy_vol,
                    fsell_vol=row.sell_vol,
                    fbuy_val=row.buy_val,
                    fsell_val=row.sell_val,
                    source="ssi_fastconnect_stream",
                )
            )

        existing_meta = self.session.exec(
            select(MarketDailyMeta)
            .where(MarketDailyMeta.symbol == row.symbol)
            .where(MarketDailyMeta.timestamp == row.ts_utc)
        ).first()
        payload = {
            "symbol": row.symbol,
            "timestamp": row.ts_utc,
            "foreign_buy_volume": row.buy_vol,
            "foreign_sell_volume": row.sell_vol,
            "foreign_buy_value": row.buy_val,
            "foreign_sell_value": row.sell_val,
            "net_foreign_volume": (row.buy_vol or 0.0) - (row.sell_vol or 0.0),
            "source": "ssi_fastconnect_stream",
        }
        if existing_meta:
            for key, value in payload.items():
                setattr(existing_meta, key, value)
            self.session.add(existing_meta)
        else:
            self.session.add(MarketDailyMeta(**payload))

    def upsert_index(self, idx: Any) -> None:
        existing = self.session.exec(
            select(IndexOHLCV)
            .where(IndexOHLCV.index_id == idx.index_id)
            .where(IndexOHLCV.timeframe == "tick")
            .where(IndexOHLCV.timestamp == idx.ts_utc)
        ).first()
        payload = {
            "index_id": idx.index_id,
            "timeframe": "tick",
            "timestamp": idx.ts_utc,
            "close": idx.index_value,
            "value": idx.total_value,
            "volume": idx.all_value,
            "source": "ssi_fastconnect_stream",
        }
        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
            self.session.add(existing)
            return
        self.session.add(IndexOHLCV(**payload))

    def upsert_bar(self, bar: Any) -> None:
        self.session.merge(
            PriceOHLCV(
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                timestamp=bar.ts_utc,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                value_vnd=float(bar.value or 0.0),
                source="ssi_fastconnect_stream",
                quality_flags={},
            )
        )

    def cleanup_dedup_older_than_days(self, days: int = 14) -> int:
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
        res = self.session.exec(delete(StreamDedup).where(StreamDedup.first_seen_at < cutoff))
        return int(getattr(res, "rowcount", 0) or 0)

    def commit(self) -> None:
        self.session.commit()

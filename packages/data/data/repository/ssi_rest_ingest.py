from __future__ import annotations

from typing import Any

from core.db.models import IndexOHLCV, MarketDailyMeta, PriceOHLCV, Ticker
from sqlmodel import Session, select


class SsiRestIngestRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_tickers(self, tickers: list[Any]) -> None:
        for item in tickers:
            tags = item.tags if isinstance(item.tags, list) else []
            row = Ticker(
                symbol=item.symbol,
                name=str(getattr(item, "sector", "") or item.symbol),
                exchange=item.exchange,
                sector=str(getattr(item, "sector", "") or ""),
                industry=str(getattr(item, "industry", "") or ""),
                shares_outstanding=int(getattr(item, "listed_shares", 0) or 0),
                tags={"tags": tags, "isin": getattr(item, "isin", None), "sectype": getattr(item, "sectype", None)},
            )
            self.session.merge(row)

    def upsert_prices_ohlcv(self, bars: list[Any], source: str) -> None:
        for bar in bars:
            row = PriceOHLCV(
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                timestamp=bar.ts_utc,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                value_vnd=float(bar.value or 0.0),
                source=source,
                quality_flags={},
            )
            self.session.merge(row)

    def upsert_index_ohlcv(self, rows: list[dict[str, Any]]) -> None:
        for item in rows:
            existing = self.session.exec(
                select(IndexOHLCV)
                .where(IndexOHLCV.index_id == item["index_id"])
                .where(IndexOHLCV.timeframe == item["timeframe"])
                .where(IndexOHLCV.timestamp == item["timestamp"])
            ).first()
            if existing:
                for key, value in item.items():
                    setattr(existing, key, value)
                self.session.add(existing)
            else:
                self.session.add(IndexOHLCV(**item))

    def upsert_market_daily_meta(self, rows: list[dict[str, Any]]) -> None:
        for item in rows:
            existing = self.session.exec(
                select(MarketDailyMeta)
                .where(MarketDailyMeta.symbol == item["symbol"])
                .where(MarketDailyMeta.timestamp == item["timestamp"])
            ).first()
            payload = {
                "symbol": item["symbol"],
                "timestamp": item["timestamp"],
                "ref_price": item.get("ref_price"),
                "ceiling_price": item.get("ceiling_price"),
                "floor_price": item.get("floor_price"),
                "source": item.get("source", "ssi_fastconnect_rest"),
            }
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
                self.session.add(existing)
            else:
                self.session.add(MarketDailyMeta(**payload))

    def commit(self) -> None:
        self.session.commit()

from __future__ import annotations

from core.db.models import IndexOHLCV, MarketDailyMeta, PriceOHLCV, Ticker as DbTicker
from data.providers.ssi_fastconnect.mapper_rest import map_daily_index, map_daily_stock_price
from data.schemas.canonical_models import Ticker
from data.repository.ssi_rest_ingest import SsiRestIngestRepository
from sqlmodel import SQLModel, Session, create_engine, select


def test_ssi_rest_ingest_idempotent() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    ticker = Ticker(symbol="FPT", exchange="HOSE", sector="Tech", listed_shares=1_000_000)
    stock_payload = [
        {
            "Tradingdate": "10/01/2025",
            "Symbol": "FPT",
            "Openprice": "100",
            "Highestprice": "110",
            "Lowestprice": "95",
            "Closeprice": "105",
            "Totalmatchvol": "1000",
            "Totalmatchval": "100000",
        }
    ]
    index_payload = [
        {
            "Indexcode": "VNINDEX",
            "TradingDate": "10/01/2025",
            "IndexValue": "1200",
            "TotalTrade": "1000000",
            "Time": "15:00:00",
        }
    ]

    bars, meta = map_daily_stock_price(stock_payload, source="ssi_fastconnect_rest")
    idx_rows = map_daily_index(index_payload, source="ssi_fastconnect_rest")

    with Session(engine) as session:
        repo = SsiRestIngestRepository(session)
        for _ in range(2):
            repo.upsert_tickers([ticker])
            repo.upsert_prices_ohlcv(bars, source="ssi_fastconnect_rest")
            repo.upsert_index_ohlcv(idx_rows)
            repo.upsert_market_daily_meta(meta)
            repo.commit()

        assert len(session.exec(select(DbTicker)).all()) == 1
        assert len(session.exec(select(PriceOHLCV)).all()) == 1
        assert len(session.exec(select(IndexOHLCV)).all()) == 1
        assert len(session.exec(select(MarketDailyMeta)).all()) == 1

from __future__ import annotations

import asyncio
import datetime as dt
import os

from core.db.session import create_db_and_tables, get_engine
from data.providers.ssi_fastconnect.provider_rest import SsiRestProvider
from data.repository.ssi_rest_ingest import SsiRestIngestRepository
from sqlmodel import Session


async def _run() -> None:
    if os.getenv("DEV_MODE", "true").lower() != "true":
        raise RuntimeError("ssi_rest_smoke.py only runs when DEV_MODE=true")

    required = ["SSI_FCDATA_BASE_URL", "SSI_CONSUMER_ID", "SSI_CONSUMER_SECRET", "SSI_PRIVATE_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env for SSI smoke: {', '.join(missing)}")

    database_url = os.getenv("DATABASE_URL", "sqlite:///./vn_invest.db")
    create_db_and_tables(database_url)
    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    symbols = ["FPT", "VCB", "HPG"]

    with Session(get_engine(database_url)) as session:
        repo = SsiRestIngestRepository(session)
        provider = SsiRestProvider(session=session)

        tickers = await provider.get_tickers()
        repo.upsert_tickers(tickers)

        for symbol in symbols:
            daily = await provider.get_daily_ohlcv(symbol, start, end)
            repo.upsert_prices_ohlcv(daily, source="ssi_fastconnect_rest")
            stock_bars, stock_meta = await provider.get_daily_stock_price(symbol, start, end)
            repo.upsert_prices_ohlcv(stock_bars, source="ssi_fastconnect_rest")
            repo.upsert_market_daily_meta(stock_meta)

        index_rows = await provider.get_daily_index("VNINDEX", start, end)
        repo.upsert_index_ohlcv(index_rows)
        repo.commit()


if __name__ == "__main__":
    asyncio.run(_run())

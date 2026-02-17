from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import json

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

from .client import SsiRestClient, fmt_date
from .mapper_rest import map_daily_index, map_daily_stock_price, map_ohlcv_rows, map_tickers
from .token import SsiTokenManager


class SsiRestProvider:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        bronze_dir: str | None = None,
        client: SsiRestClient | None = None,
    ) -> None:
        base = base_url or os.getenv("SSI_FCDATA_BASE_URL", "")
        if not base:
            raise RuntimeError("SSI_FCDATA_BASE_URL is required for SSI REST provider")
        self.base_url = base.rstrip("/")
        self.bronze_dir = Path(bronze_dir or os.getenv("BRONZE_LAKE_DIR", "data_lake/bronze"))
        self.source = "ssi_fastconnect_rest"

        if client is None:
            token_manager = SsiTokenManager(base_url=self.base_url)
            client = SsiRestClient(base_url=self.base_url, token_manager=token_manager)
        self.client = client

    async def get_tickers(self) -> list[Any]:
        securities = await self.client.request("GET", "/Securities")
        await self._log_bronze("securities", securities)

        details_resp = await self.client.request("GET", "/SecuritiesDetails")
        await self._log_bronze("securities_details", details_resp)

        details = details_resp.get("RepeatedInfo") if isinstance(details_resp, dict) else details_resp
        if not isinstance(details, list):
            raise ValueError("SecuritiesDetails response missing RepeatedInfo list")
        if not isinstance(securities, list):
            raise ValueError("Securities response must be a JSON list")
        return map_tickers(securities, details)

    async def get_daily_ohlcv(
        self, symbol: str, start: dt.date | dt.datetime, end: dt.date | dt.datetime
    ) -> list[Any]:
        payload = await self.client.request(
            "GET",
            "/DailyOhlc",
            params={"symbol": symbol, "from": fmt_date(start), "to": fmt_date(end)},
        )
        await self._log_bronze("daily_ohlcv", payload)
        if not isinstance(payload, list):
            raise ValueError("DailyOhlc response must be a JSON list")
        return map_ohlcv_rows(payload, timeframe="1D", source=self.source)

    async def get_intraday_ohlcv(
        self,
        symbol: str,
        start: dt.date | dt.datetime,
        end: dt.date | dt.datetime,
        resolution: str,
    ) -> list[Any]:
        payload = await self.client.request(
            "GET",
            "/IntradayOhlc",
            params={
                "symbol": symbol,
                "from": fmt_date(start),
                "to": fmt_date(end),
                "resolution": resolution,
            },
        )
        await self._log_bronze("intraday_ohlcv", payload)
        if not isinstance(payload, list):
            raise ValueError("IntradayOhlc response must be a JSON list")
        return map_ohlcv_rows(payload, timeframe=resolution, source=self.source)

    async def get_daily_index(
        self, index_id: str, start: dt.date | dt.datetime, end: dt.date | dt.datetime
    ) -> list[dict[str, Any]]:
        payload = await self.client.request(
            "GET",
            "/DailyIndex",
            params={"indexId": index_id, "from": fmt_date(start), "to": fmt_date(end)},
        )
        await self._log_bronze("daily_index", payload)
        if not isinstance(payload, list):
            raise ValueError("DailyIndex response must be a JSON list")
        return map_daily_index(payload, source=self.source)

    async def get_daily_stock_price(
        self, symbol: str, start: dt.date | dt.datetime, end: dt.date | dt.datetime
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        payload = await self.client.request(
            "GET",
            "/DailyStockPrice",
            params={"symbol": symbol, "from": fmt_date(start), "to": fmt_date(end)},
        )
        await self._log_bronze("daily_stock_price", payload)
        if not isinstance(payload, list):
            raise ValueError("DailyStockPrice response must be a JSON list")
        return map_daily_stock_price(payload, source=self.source)

    async def _log_bronze(self, endpoint: str, payload: Any) -> None:
        channel = f"ssi_rest_{endpoint}"
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y/%m/%d")
        out_dir = self.bronze_dir / "ssi_fastconnect" / channel / day
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "events.jsonl"
        rec = {"channel": channel, "payload": payload, "received_at_utc": now}
        with out_file.open("ab") as f:
            if orjson is not None:
                f.write(orjson.dumps(rec) + b"\n")
            else:
                f.write((json.dumps(rec, ensure_ascii=False, default=str) + "\n").encode("utf-8"))

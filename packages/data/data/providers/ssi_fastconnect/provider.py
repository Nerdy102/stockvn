from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from data.providers.base import BaseMarketDataProvider


class SsiFastConnectProvider(BaseMarketDataProvider):
    """SSI FastConnect provider stub for offline-safe operation."""

    def __init__(self) -> None:
        self.base_url = os.getenv("SSI_FCDATA_BASE_URL", "https://fc-data.ssi.com.vn/api/v2")
        self.consumer_id = os.getenv("SSI_CONSUMER_ID", os.getenv("consumerID", ""))
        self.consumer_secret = os.getenv("SSI_CONSUMER_SECRET", os.getenv("consumerSecret", ""))
        self.private_key_path = os.getenv("SSI_PRIVATE_KEY_PATH", os.getenv("privateKeyPath", ""))
        self.access_token = os.getenv("SSI_ACCESS_TOKEN", "")
        self.token_cache_file = os.getenv("SSI_TOKEN_CACHE_FILE", "")

    def _ensure(self) -> None:
        if not (self.consumer_id and self.consumer_secret and self.private_key_path):
            raise RuntimeError(
                "Missing SSI credentials in env (.env): SSI_CONSUMER_ID/SECRET/PRIVATE_KEY_PATH"
            )

    def get_tickers(self) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_intraday(
        self,
        symbol: str,
        timeframe: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_fundamentals(self) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_fundamentals_stub(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["symbol", "period_end", "public_date", "as_of_date", "statement_type"]
        )

    def stream_messages_stub(self) -> None:
        """Streaming stub for SSI websocket/channel integration (disabled by default)."""
        self._ensure()
        raise NotImplementedError


    def set_access_token(self, token: str) -> None:
        self.access_token = token
        if self.token_cache_file:
            try:
                with open(self.token_cache_file, "w", encoding="utf-8") as f:
                    f.write(token)
            except OSError:
                return

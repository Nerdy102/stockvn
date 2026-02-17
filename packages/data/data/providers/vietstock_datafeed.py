from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from data.providers.base import BaseMarketDataProvider


class VietstockDataFeedProvider(BaseMarketDataProvider):
    """Stub provider for Vietstock DataFeed (commercial data).

    - KHÔNG hardcode API key.
    - KHÔNG nhúng URL endpoints.
    - Bạn cần triển khai theo tài liệu & điều khoản của nhà cung cấp.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("VIETSTOCK_API_KEY", "")

    def _ensure(self) -> None:
        if not self.api_key:
            raise RuntimeError("Missing VIETSTOCK_API_KEY in env (.env).")

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

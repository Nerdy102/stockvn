from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import pandas as pd

from data.providers.base import BaseMarketDataProvider


class SsiFastConnectProvider(BaseMarketDataProvider):
    """Stub provider for SSI FastConnect (commercial API).

    - KHÔNG hardcode secrets.
    - KHÔNG nhúng URL endpoints.
    - Bạn cần triển khai theo tài liệu & hợp đồng sử dụng dữ liệu của SSI.
    """

    def __init__(self) -> None:
        self.username = os.getenv("SSI_USERNAME", "")
        self.password = os.getenv("SSI_PASSWORD", "")
        self.api_key = os.getenv("SSI_API_KEY", "")

    def _ensure(self) -> None:
        if not (self.username and self.password and self.api_key):
            raise RuntimeError("Missing SSI credentials in env (.env).")

    def get_tickers(self) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_ohlcv(self, symbol: str, timeframe: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_intraday(self, symbol: str, timeframe: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

    def get_fundamentals(self) -> pd.DataFrame:
        self._ensure()
        raise NotImplementedError

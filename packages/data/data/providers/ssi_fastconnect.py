from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import pandas as pd

from data.providers.base import BaseMarketDataProvider


class SsiFastConnectProvider(BaseMarketDataProvider):
    """SSI FastConnect scaffold (provider-agnostic architecture keeps mapper/repo separate)."""

    def __init__(self) -> None:
        self.consumer_id = os.getenv("SSI_CONSUMER_ID", "")
        self.consumer_secret = os.getenv("SSI_CONSUMER_SECRET", "")
        self.private_key_path = os.getenv("SSI_PRIVATE_KEY_PATH", "")
        self.access_token = os.getenv("SSI_ACCESS_TOKEN", "")

    def _ensure(self) -> None:
        if not (self.consumer_id and self.consumer_secret and self.private_key_path):
            raise RuntimeError("Missing SSI credentials in env (.env): SSI_CONSUMER_ID/SECRET/PRIVATE_KEY_PATH")

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

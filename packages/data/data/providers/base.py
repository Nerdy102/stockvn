from __future__ import annotations

import abc
import datetime as dt
from typing import Optional

import pandas as pd


class BaseMarketDataProvider(abc.ABC):
    """Base interface for market data providers."""

    @abc.abstractmethod
    def get_tickers(self) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def get_intraday(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def get_fundamentals(self) -> pd.DataFrame:
        raise NotImplementedError

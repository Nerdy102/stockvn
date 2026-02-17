from __future__ import annotations

import abc
import datetime as dt

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
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def get_intraday(
        self,
        symbol: str,
        timeframe: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def get_fundamentals(self) -> pd.DataFrame:
        raise NotImplementedError

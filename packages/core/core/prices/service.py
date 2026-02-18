from __future__ import annotations

import datetime as dt

import pandas as pd

from core.corporate_actions import adjust_prices
from data.providers.base import BaseMarketDataProvider


class PriceService:
    def __init__(
        self,
        provider: BaseMarketDataProvider,
        corporate_actions: pd.DataFrame | None = None,
    ) -> None:
        self.provider = provider
        self.corporate_actions = corporate_actions

    def get_ohlcv(
        self,
        symbol: str,
        start: dt.date | None,
        end: dt.date | None,
        timeframe: str,
        adjusted: bool,
        total_return: bool,
        *,
        as_of_date: dt.date | None = None,
    ) -> pd.DataFrame:
        bars = self.provider.get_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        if bars.empty:
            return bars
        return adjust_prices(
            symbol=symbol,
            bars=bars,
            start=start or dt.date.min,
            end=end or dt.date.max,
            method=("ca" if adjusted else "none"),
            corporate_actions=self.corporate_actions,
            as_of_date=as_of_date,
            total_return=total_return,
        )

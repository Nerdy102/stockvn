from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from data.schemas.ssi_fcdata import VN_TZ, DailyOhlcRecord, DailyStockPriceRecord


@dataclass
class CanonicalPriceBar:
    symbol: str
    timeframe: str
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    value: float | None = None
    source: str = "ssi_fcdata"
    quality_flags: list[str] = field(default_factory=list)


class SSIMapper:
    @staticmethod
    def _to_utc(ts: dt.datetime | None, trading_date: dt.date) -> dt.datetime:
        base = ts or dt.datetime.combine(trading_date, dt.time(0, 0), tzinfo=VN_TZ)
        if base.tzinfo is None:
            base = base.replace(tzinfo=VN_TZ)
        return base.astimezone(dt.timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _flags_for_invariants(
        open_px: float, high: float, low: float, close: float, volume: float
    ) -> list[str]:
        flags: list[str] = []
        if min(open_px, high, low, close, volume) < 0:
            flags.append("negative_value")
        if high < max(open_px, close) or low > min(open_px, close):
            flags.append("suspicious_ohlc")
        return flags

    @classmethod
    def map_daily_ohlc(cls, payload: dict[str, Any], timeframe: str = "1D") -> CanonicalPriceBar:
        row = DailyOhlcRecord.model_validate(payload)
        row.record_schema_drift_metrics()
        flags = cls._flags_for_invariants(row.Open, row.High, row.Low, row.Close, row.Volume)
        if row.Value is None:
            flags.append("missing_value")
        return CanonicalPriceBar(
            symbol=row.Symbol,
            timeframe=timeframe,
            ts=cls._to_utc(row.Time, row.TradingDate),
            open=row.Open,
            high=row.High,
            low=row.Low,
            close=row.Close,
            volume=row.Volume,
            value=row.Value,
            quality_flags=flags,
        )

    @classmethod
    def map_daily_stock_price(cls, payload: dict[str, Any]) -> CanonicalPriceBar:
        row = DailyStockPriceRecord.model_validate(payload)
        row.record_schema_drift_metrics()
        open_px = row.Openprice or 0.0
        high = row.Highestprice or 0.0
        low = row.Lowestprice or 0.0
        close = row.Closeprice or 0.0
        volume = row.Totalmatchvol or 0.0
        flags = cls._flags_for_invariants(open_px, high, low, close, volume)
        if row.Closeprice is None or row.Openprice is None:
            flags.append("missing_fields")
        return CanonicalPriceBar(
            symbol=row.Symbol,
            timeframe="1D",
            ts=cls._to_utc(row.Time, row.Tradingdate),
            open=open_px,
            high=high,
            low=low,
            close=close,
            volume=volume,
            value=row.Totalmatchval,
            quality_flags=flags,
        )

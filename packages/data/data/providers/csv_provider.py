from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from data.providers.base import BaseMarketDataProvider


class CsvProvider(BaseMarketDataProvider):
    """Offline CSV provider for demo (synthetic)."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def get_tickers(self) -> pd.DataFrame:
        df = pd.read_csv(self.base_dir / "tickers_demo.csv")
        df["is_bank"] = df["is_bank"].astype(bool)
        df["is_broker"] = df["is_broker"].astype(bool)
        return df

    def get_fundamentals(self) -> pd.DataFrame:
        df = pd.read_csv(self.base_dir / "fundamentals_demo.csv")
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        df["is_bank"] = df["is_bank"].astype(bool)
        df["is_broker"] = df["is_broker"].astype(bool)
        return df

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        if timeframe in {"15m", "60m"}:
            return self.get_intraday(symbol, timeframe, start, end)

        df = pd.read_csv(self.base_dir / "prices_demo_1d.csv")
        df = df[df["symbol"] == symbol].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        df = df.sort_values("date")
        return df[["date", "open", "high", "low", "close", "volume", "value_vnd"]]

    def get_intraday(
        self,
        symbol: str,
        timeframe: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic intraday bars from daily bars (MVP).

        Note: Intraday data is generated deterministically per (symbol,timeframe) so it's stable offline.
        """
        daily = self.get_ohlcv(symbol, "1D", start, end).copy()
        if daily.empty:
            return daily

        minutes = 15 if timeframe == "15m" else 60
        rr = np.random.default_rng(abs(hash((symbol, timeframe))) % (2**32))

        rows = []
        for _, r in daily.iterrows():
            day: dt.date = r["date"]
            o, h, low_px, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
            v = float(r.get("volume", 0.0))

            # Continuous sessions only (simplified)
            t1 = dt.datetime.combine(day, dt.time(9, 15))
            t2 = dt.datetime.combine(day, dt.time(11, 30))
            t3 = dt.datetime.combine(day, dt.time(13, 0))
            t4 = dt.datetime.combine(day, dt.time(14, 30))

            def gen_times(a: dt.datetime, b: dt.datetime) -> list[dt.datetime]:
                out = []
                cur = a
                while cur < b:
                    out.append(cur)
                    cur += dt.timedelta(minutes=minutes)
                return out

            times = gen_times(t1, t2) + gen_times(t3, t4)
            n = len(times)
            if n == 0:
                continue

            path = np.linspace(o, c, n)
            noise_scale = max(0.1, (h - low_px) * 0.02)
            close_path = np.clip(path + rr.normal(0, noise_scale, n), low_px, h)

            vol_path = rr.uniform(0.03, 0.10, n)
            vol_path = vol_path / vol_path.sum() * max(0.0, v)

            prev = o
            for i, ts in enumerate(times):
                bar_close = float(close_path[i])
                bar_open = float(prev)
                bar_high = float(max(bar_open, bar_close) * (1.0 + abs(rr.normal(0, 0.002))))
                bar_low = float(min(bar_open, bar_close) * (1.0 - abs(rr.normal(0, 0.002))))
                bar_vol = float(vol_path[i])

                rows.append(
                    {
                        "timestamp": ts,
                        "open": bar_open,
                        "high": bar_high,
                        "low": bar_low,
                        "close": bar_close,
                        "volume": bar_vol,
                        "value_vnd": bar_close * bar_vol,
                    }
                )
                prev = bar_close

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values("timestamp")
        return out

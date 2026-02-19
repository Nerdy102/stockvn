from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from data.providers.base import BaseMarketDataProvider


class CryptoPublicOHLCVProvider(BaseMarketDataProvider):
    def __init__(
        self,
        cache_dir: str | Path = "artifacts/cache/crypto_public",
        exchange: str = "binance_public",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exchange = exchange
        self.api_base = "https://api.binance.com/api/v3"
        self.symbol_alias = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "SOL": "SOLUSDT"}
        self.offline_file = Path("data_demo/crypto_prices_demo_1d.csv")

    def get_tickers(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"symbol": "BTCUSDT", "exchange": self.exchange, "name": "Bitcoin"},
                {"symbol": "ETHUSDT", "exchange": self.exchange, "name": "Ethereum"},
            ]
        )

    def get_fundamentals(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "note"])

    def _norm_symbol(self, symbol: str) -> str:
        s = str(symbol).upper().strip()
        if s in self.symbol_alias:
            return self.symbol_alias[s]
        return s

    def _interval(self, timeframe: str) -> str:
        return "1d" if timeframe == "1D" else "1h"

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_dir / f"{self.exchange}_{symbol}_{timeframe}.json"

    def _read_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame()

    def _write_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, timeframe)
        path.write_text(df.to_json(orient="records", force_ascii=False), encoding="utf-8")

    def _fetch_remote(self, symbol: str, timeframe: str) -> pd.DataFrame:
        interval = self._interval(timeframe)
        url = f"{self.api_base}/klines?symbol={symbol}&interval={interval}&limit=756"
        backoff = [0.3, 0.8, 1.5]
        err = None
        for delay in backoff:
            try:
                req = Request(url, headers={"User-Agent": "stockvn-crypto-provider/1.0"})
                with urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                rows = []
                for r in data:
                    ts = dt.datetime.utcfromtimestamp(int(r[0]) / 1000)
                    rows.append(
                        {
                            "date": ts.date().isoformat() if timeframe == "1D" else ts.isoformat(),
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "close": float(r[4]),
                            "volume": float(r[5]),
                            "value_vnd": float(r[4]) * float(r[5]),
                        }
                    )
                out = pd.DataFrame(rows)
                if timeframe == "1D":
                    out["date"] = pd.to_datetime(out["date"]).dt.date
                else:
                    out = out.rename(columns={"date": "timestamp"})
                    out["timestamp"] = pd.to_datetime(out["timestamp"])
                return out
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(delay)
        raise RuntimeError(
            f"Không lấy được dữ liệu — hãy dùng demo offline hoặc data_drop. Lỗi: {err}"
        )

    def _offline_fallback(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.offline_file.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.offline_file)
        df = df[df["symbol"] == symbol].copy()
        if timeframe == "1D":
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df[["date", "open", "high", "low", "close", "volume", "value_vnd"]]
        # 60m synthetic from 1d
        rows: list[dict[str, float | dt.datetime]] = []
        for _, r in df.iterrows():
            base = pd.Timestamp(r["date"])
            for h in [0, 6, 12, 18]:
                ts = base + pd.Timedelta(hours=h)
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r["volume"]) / 4.0,
                        "value_vnd": float(r["value_vnd"]) / 4.0,
                    }
                )
        return pd.DataFrame(rows)

    def get_ohlcv(
        self, symbol: str, timeframe: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> pd.DataFrame:
        sym = self._norm_symbol(symbol)
        try:
            df = self._fetch_remote(sym, timeframe)
            self._write_cache(sym, timeframe, df)
        except RuntimeError:
            df = self._read_cache(sym, timeframe)
            if df.empty:
                df = self._offline_fallback(sym, timeframe)
        if df.empty:
            return df
        if timeframe == "1D" and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]
        return df

    def get_intraday(
        self, symbol: str, timeframe: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> pd.DataFrame:
        return self.get_ohlcv(symbol, timeframe, start, end)

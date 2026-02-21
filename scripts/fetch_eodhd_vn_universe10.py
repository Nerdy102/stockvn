from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import zstandard as zstd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_UNIVERSE = ["FPT", "HPG", "SCS", "SSI", "ACV", "MWG", "HHV", "VCG", "VCB", "VNM"]
EODHD_BASE_URL = "https://eodhd.com/api"
CACHE_DIR = Path("artifacts/eodhd_cache")

log = logging.getLogger(__name__)


def _load_api_token() -> str:
    token = os.getenv("EODHD_API_TOKEN", "").strip()
    if token:
        return token
    try:
        from configs.secrets_local import EODHD_API_TOKEN  # type: ignore

        token = str(EODHD_API_TOKEN).strip()
    except Exception as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            "Missing EODHD API token. Set EODHD_API_TOKEN env var or create configs/secrets_local.py"
        ) from exc
    if not token:
        raise RuntimeError(
            "EODHD API token is empty. Set EODHD_API_TOKEN env var or configs/secrets_local.EODHD_API_TOKEN"
        )
    return token


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _get_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload type at {url}: {type(payload)}")
    return payload


def _default_from_to() -> tuple[str, str]:
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=365)
    return from_date.isoformat(), to_date.isoformat()


def _cache_file(symbol: str, from_date: str, to_date: str) -> Path:
    safe = symbol.replace(".", "_")
    return CACHE_DIR / f"{safe}_{from_date}_{to_date}.json.zst"


def _read_cached_json(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    text = zstd.ZstdDecompressor().decompress(raw).decode("utf-8")
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"Invalid cache payload in {path}")
    return payload


def _write_cached_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstd.ZstdCompressor(level=6).compress(json.dumps(payload).encode("utf-8"))
    path.write_bytes(compressed)


def fetch_symbol_list(api_token: str) -> set[str]:
    rows = _get_json(
        f"{EODHD_BASE_URL}/exchange-symbol-list/VN",
        params={"api_token": api_token, "fmt": "json"},
    )
    df = pd.DataFrame(rows)
    if "Code" not in df.columns:
        raise ValueError("EODHD exchange-symbol-list response missing Code column")
    codes = df["Code"].astype(str).str.upper().str.strip()
    return {f"{code}.VN" for code in codes if code}


def fetch_eod_for_symbol(
    symbol_with_exchange: str,
    api_token: str,
    from_date: str,
    to_date: str,
    refresh: bool,
) -> list[dict[str, Any]]:
    cache_path = _cache_file(symbol_with_exchange, from_date, to_date)
    if cache_path.exists() and not refresh:
        return _read_cached_json(cache_path)

    payload = _get_json(
        f"{EODHD_BASE_URL}/eod/{symbol_with_exchange}",
        params={
            "api_token": api_token,
            "from": from_date,
            "to": to_date,
            "period": "d",
            "fmt": "json",
        },
    )
    _write_cached_json(cache_path, payload)
    return payload


def _normalize_prices(raw_rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(raw_rows)
    expected = ["date", "open", "high", "low", "close", "volume"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} for {symbol}")
    out = df[expected].copy()
    out.insert(0, "symbol", symbol)
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    return out


def _build_tickers(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "exchange": "VN",
                "sector": "UNKNOWN",
                "industry": "UNKNOWN",
                "shares_outstanding": 0,
                "market_cap": 0,
                "is_bank": False,
                "is_broker": symbol == "SSI",
                "tags": "VN10_EODHD",
            }
        )
    return pd.DataFrame(rows)


def run(
    from_date: str,
    to_date: str,
    refresh: bool,
    out_prices: Path,
    out_tickers: Path,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api_token = _load_api_token()
    listed = fetch_symbol_list(api_token)

    all_prices: list[pd.DataFrame] = []
    valid_symbols: list[str] = []
    for symbol in DEFAULT_UNIVERSE:
        symbol_with_exchange = f"{symbol}.VN"
        if symbol_with_exchange not in listed:
            hint = difflib.get_close_matches(symbol_with_exchange, sorted(listed), n=1)
            log.warning("Symbol %s not found in VN listing. hint=%s", symbol_with_exchange, hint)
            continue
        raw_rows = fetch_eod_for_symbol(
            symbol_with_exchange, api_token, from_date, to_date, refresh
        )
        if not raw_rows:
            log.warning(
                "No data for %s in range %s..%s (free-plan/history limit?)",
                symbol,
                from_date,
                to_date,
            )
            continue
        df_symbol = _normalize_prices(raw_rows, symbol)
        all_prices.append(df_symbol)
        valid_symbols.append(symbol)

    if not all_prices:
        raise RuntimeError("No EOD rows fetched for any VN10 symbols")

    prices = pd.concat(all_prices, ignore_index=True)
    prices["value_vnd"] = prices["close"].astype(float) * prices["volume"].astype(float)
    prices = prices[
        ["symbol", "date", "open", "high", "low", "close", "volume", "value_vnd"]
    ].drop_duplicates(subset=["symbol", "date"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    out_prices.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_prices, index=False)

    tickers = _build_tickers(valid_symbols)
    out_tickers.parent.mkdir(parents=True, exist_ok=True)
    tickers.to_csv(out_tickers, index=False)

    log.info("Wrote %s rows to %s", len(prices), out_prices)
    log.info("Wrote %s symbols to %s", len(tickers), out_tickers)


def parse_args() -> argparse.Namespace:
    default_from, default_to = _default_from_to()
    parser = argparse.ArgumentParser(
        description="Fetch VN10 OHLCV from EODHD and export CSV demo files"
    )
    parser.add_argument("--from", dest="from_date", default=os.getenv("FROM", default_from))
    parser.add_argument("--to", dest="to_date", default=os.getenv("TO", default_to))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-prices", default="data_demo/prices_demo_1d.csv")
    parser.add_argument("--out-tickers", default="data_demo/tickers_demo.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        from_date=args.from_date,
        to_date=args.to_date,
        refresh=bool(args.refresh),
        out_prices=Path(args.out_prices),
        out_tickers=Path(args.out_tickers),
    )


if __name__ == "__main__":
    main()

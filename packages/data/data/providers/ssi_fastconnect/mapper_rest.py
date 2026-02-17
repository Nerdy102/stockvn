from __future__ import annotations

import datetime as dt
from typing import Any

from data.schemas.canonical_models import Bar, Ticker

VN_TZ = dt.timezone(dt.timedelta(hours=7))
UTC = dt.timezone.utc


def _pick(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _to_float(v: Any) -> float | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).replace(",", "").strip())


def _to_int(v: Any) -> int | None:
    fv = _to_float(v)
    return int(fv) if fv is not None else None


def _require_float(d: dict[str, Any], *keys: str, context: str) -> float:
    val = _to_float(_pick(d, *keys))
    if val is None:
        raise ValueError(f"Missing required numeric field {keys} in {context}")
    return float(val)


def parse_vn_ts_utc(raw_date: Any, raw_time: Any) -> dt.datetime:
    if raw_date in (None, ""):
        raise ValueError("Missing TradingDate/Tradingdate in SSI REST payload")
    d = str(raw_date).strip()
    day: dt.date
    if "/" in d:
        day = dt.datetime.strptime(d, "%d/%m/%Y").date()
    else:
        day = dt.date.fromisoformat(d)

    if raw_time in (None, ""):
        local = dt.datetime.combine(day, dt.time(0, 0, 0), tzinfo=VN_TZ)
        return local.astimezone(UTC)

    t = str(raw_time).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_t = dt.datetime.strptime(t, fmt).time()
            local = dt.datetime.combine(day, parsed_t, tzinfo=VN_TZ)
            return local.astimezone(UTC)
        except ValueError:
            continue
    raise ValueError(f"Invalid SSI Time format: {raw_time}")


def map_tickers(securities: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[Ticker]:
    details_by_symbol = {str(item.get("Symbol", "")).upper(): item for item in details}
    out: list[Ticker] = []
    for item in securities:
        symbol = str(item.get("Symbol", "")).upper()
        if not symbol:
            raise ValueError(f"Securities record missing Symbol: {item}")
        d = details_by_symbol.get(symbol, {})
        exchange = str(_pick(d, "Exchange", "MarketID", "MarketId", "Market") or "")
        if not exchange:
            raise ValueError(f"Missing exchange for symbol={symbol} in SecuritiesDetails")

        out.append(
            Ticker(
                symbol=symbol,
                exchange=exchange,
                sector=str(item.get("StockName") or item.get("StockEnName") or ""),
                instrument_type=str(_pick(d, "SecType") or "stock").lower(),
                lot_size=_to_int(_pick(d, "LotSize")),
                listed_shares=_to_int(_pick(d, "ListedShare")),
                isin=_pick(d, "Isin"),
                sectype=_pick(d, "SecType"),
                market_id=_pick(d, "MarketID", "MarketId"),
            )
        )
    return out


def map_ohlcv_rows(payload: list[dict[str, Any]], *, timeframe: str, source: str) -> list[Bar]:
    bars: list[Bar] = []
    for row in payload:
        symbol = str(_pick(row, "Symbol") or "").upper()
        if not symbol:
            raise ValueError(f"Missing Symbol in OHLCV payload: {row}")
        ts_utc = parse_vn_ts_utc(_pick(row, "TradingDate", "Tradingdate"), row.get("Time"))
        try:
            bars.append(
                Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    ts_utc=ts_utc,
                    open=_require_float(row, "Open", "Openprice", context=f"OHLCV/{symbol}"),
                    high=_require_float(row, "High", "Highestprice", context=f"OHLCV/{symbol}"),
                    low=_require_float(row, "Low", "Lowestprice", context=f"OHLCV/{symbol}"),
                    close=_require_float(row, "Close", "Closeprice", context=f"OHLCV/{symbol}"),
                    volume=_require_float(row, "Volume", "Totalmatchvol", context=f"OHLCV/{symbol}"),
                    value=_to_float(_pick(row, "Value", "Totalmatchval", "TotalTrade")),
                    data_source=source,
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid numeric fields in OHLCV payload for {symbol}: {exc}") from exc
    return bars


def map_daily_index(payload: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in payload:
        index_id = str(_pick(row, "Indexcode", "IndexCode", "index_id") or "").upper()
        if not index_id:
            raise ValueError(f"Missing Indexcode in daily index payload: {row}")
        out.append(
            {
                "index_id": index_id,
                "timeframe": "1D",
                "timestamp": parse_vn_ts_utc(_pick(row, "TradingDate", "Tradingdate"), row.get("Time")),
                "open": _to_float(row.get("Open")),
                "high": _to_float(row.get("High")),
                "low": _to_float(row.get("Low")),
                "close": _to_float(_pick(row, "IndexValue", "Close")),
                "value": _to_float(_pick(row, "TotalTrade", "Totalmatchval")),
                "volume": _to_float(row.get("Totalmatchvol")),
                "source": source,
            }
        )
    return out


def map_daily_stock_price(payload: list[dict[str, Any]], source: str) -> tuple[list[Bar], list[dict[str, Any]]]:
    bars = map_ohlcv_rows(payload, timeframe="1D", source=source)
    meta: list[dict[str, Any]] = []
    for row, bar in zip(payload, bars):
        meta.append(
            {
                "symbol": bar.symbol,
                "timestamp": bar.ts_utc,
                "ref_price": _to_float(_pick(row, "Ref", "RefPrice")),
                "ceiling_price": _to_float(_pick(row, "Ceiling", "CeilingPrice")),
                "floor_price": _to_float(_pick(row, "Floor", "FloorPrice")),
                "foreign_buy_volume": _to_float(_pick(row, "Foreignbuyvoltotal", "ForeignBuyVolTotal")),
                "foreign_sell_volume": _to_float(_pick(row, "Foreignsellvoltotal", "ForeignSellVolTotal")),
                "foreign_buy_value": _to_float(_pick(row, "Foreignbuyvaltotal", "ForeignBuyValTotal")),
                "foreign_sell_value": _to_float(
                    _pick(row, "Foreignsellvaltotal", "Toreignsellvaltotal", "ForeignSellValTotal")
                ),
                "net_foreign_volume": _to_float(_pick(row, "Netforeignvol", "Netforeivol")),
                "source": source,
            }
        )
    return bars, meta

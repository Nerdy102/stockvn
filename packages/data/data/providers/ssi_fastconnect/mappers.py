from __future__ import annotations

import datetime as dt
from typing import Any

from data.schemas.canonical_models import (
    Bar,
    ForeignRoomSnapshot,
    IndexSnapshot,
    OddLotSnapshot,
    QuoteLevel,
    TradePrint,
)

VN = dt.timezone(dt.timedelta(hours=7))

FIELD_ALIAS_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "common": {
        "trading_date": ("TradingDate", "Tradingdate"),
        "market_id": ("MarketID", "MarketId"),
        "avg_price": ("Avg", "AvgPrice", "AveragePrice"),
        "prior_close": ("PriorVal",),
        "ceiling_price": ("Ceiling", "CeilingPrice"),
        "floor_price": ("Floor", "FloorPrice"),
        "ref_price": ("Ref", "RefPrice"),
    },
}


def _pick(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _num(v: Any) -> float | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).replace(",", "").strip())


def _parse_dt(raw: dict[str, Any]) -> dt.datetime:
    d = str(_pick(raw, "TradingDate", "Tradingdate") or "01/01/1970")
    t = str(_pick(raw, "Time", "TradingTime") or "00:00:00")
    day = dt.datetime.strptime(d, "%d/%m/%Y").date() if "/" in d else dt.date.fromisoformat(d)
    if " " in t:
        ts = dt.datetime.fromisoformat(t).replace(tzinfo=VN)
        return ts.astimezone(dt.timezone.utc)
    if len(t) == 6 and t.isdigit():
        t = f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    ts = dt.datetime.combine(day, dt.time.fromisoformat(t), tzinfo=VN)
    return ts.astimezone(dt.timezone.utc)


def map_daily_ohlc(resp: list[dict[str, Any]]) -> list[Bar]:
    out: list[Bar] = []
    for r in resp:
        out.append(
            Bar(
                symbol=str(r["Symbol"]),
                timeframe="1D",
                ts_utc=_parse_dt(r),
                open=float(_num(r.get("Open")) or 0.0),
                high=float(_num(r.get("High")) or 0.0),
                low=float(_num(r.get("Low")) or 0.0),
                close=float(_num(r.get("Close")) or 0.0),
                volume=float(_num(r.get("Volume")) or 0.0),
                value=_num(r.get("Value")),
                data_source="ssi_fastconnect",
            )
        )
    return out


def map_intraday_ohlc(resp: list[dict[str, Any]]) -> list[Bar]:
    out = map_daily_ohlc(resp)
    for b in out:
        b.timeframe = "1m"
    return out


def map_stream_x_quote(msg: dict[str, Any]) -> QuoteLevel:
    bids = [float(_num(msg.get(f"BidPrice{i}")) or 0.0) for i in range(1, 11)]
    asks = [float(_num(msg.get(f"AskPrice{i}")) or 0.0) for i in range(1, 11)]
    bidv = [float(_num(msg.get(f"BidVol{i}")) or 0.0) for i in range(1, 11)]
    askv = [float(_num(msg.get(f"AskVol{i}")) or 0.0) for i in range(1, 11)]
    return QuoteLevel(
        symbol=str(msg.get("Symbol", "")),
        ts_utc=_parse_dt(msg),
        bid_prices=bids,
        ask_prices=asks,
        bid_volumes=bidv,
        ask_volumes=askv,
    )


def map_stream_x_trade(msg: dict[str, Any]) -> TradePrint:
    return TradePrint(
        symbol=str(msg.get("Symbol", "")),
        ts_utc=_parse_dt(msg),
        last_price=float(_num(_pick(msg, "LastPrice")) or 0.0),
        last_vol=float(_num(_pick(msg, "LastVol")) or 0.0),
        total_val=_num(_pick(msg, "TotalVal")),
        total_vol=_num(_pick(msg, "TotalVol")),
        side=str(_pick(msg, "Side")) if _pick(msg, "Side") is not None else None,
    )


def map_stream_x(msg: dict[str, Any]) -> tuple[QuoteLevel, TradePrint]:
    return map_stream_x_quote(msg), map_stream_x_trade(msg)


def map_stream_r(msg: dict[str, Any]) -> ForeignRoomSnapshot:
    return ForeignRoomSnapshot(
        symbol=str(msg.get("Symbol", "")),
        ts_utc=_parse_dt(msg),
        total_room=_num(_pick(msg, "TotalRoom", "FTotalRoom")),
        current_room=_num(_pick(msg, "CurrentRoom", "FCurrentRoom")),
        buy_vol=_num(_pick(msg, "BuyVol", "FBuyVol")),
        sell_vol=_num(_pick(msg, "SellVol", "FSellVol")),
        buy_val=_num(_pick(msg, "BuyVal", "FBuyVal")),
        sell_val=_num(_pick(msg, "SellVal", "FSellVal")),
    )


def map_stream_mi(msg: dict[str, Any]) -> IndexSnapshot:
    return IndexSnapshot(
        index_id=str(_pick(msg, "IndexId", "IndexID")),
        ts_utc=_parse_dt(msg),
        index_value=float(_num(_pick(msg, "IndexValue")) or 0.0),
        change=_num(msg.get("Change")),
        ratio_change=_num(msg.get("RatioChange")),
        advances=int(_num(_pick(msg, "Advances")) or 0),
        declines=int(_num(_pick(msg, "Declines")) or 0),
        nochange=int(_num(_pick(msg, "NoChanges", "Nochange")) or 0),
        ceilings=int(_num(_pick(msg, "Ceilings")) or 0),
        floors=int(_num(_pick(msg, "Floors")) or 0),
        total_value=_num(_pick(msg, "TotalValueOd", "TotalValue")),
        all_value=_num(_pick(msg, "AllValue")),
        session=(
            str(_pick(msg, "TradingSession")) if _pick(msg, "TradingSession") is not None else None
        ),
    )


def map_stream_b(msg: dict[str, Any]) -> Bar:
    return Bar(
        symbol=str(msg.get("Symbol", "")),
        timeframe="1m",
        ts_utc=_parse_dt(msg),
        open=float(_num(msg.get("Open")) or 0.0),
        high=float(_num(msg.get("High")) or 0.0),
        low=float(_num(msg.get("Low")) or 0.0),
        close=float(_num(msg.get("Close")) or 0.0),
        volume=float(_num(msg.get("Volume")) or 0.0),
        value=_num(msg.get("Value")),
        data_source="ssi_fastconnect_stream",
    )


def map_daily_stock_price(resp: list[dict[str, Any]]) -> tuple[list[Bar], list[dict[str, Any]]]:
    bars: list[Bar] = []
    meta: list[dict[str, Any]] = []
    for r in resp:
        bars.append(
            Bar(
                symbol=str(r["Symbol"]),
                timeframe="1D",
                ts_utc=_parse_dt(r),
                open=float(_num(_pick(r, "Open", "Openprice")) or 0.0),
                high=float(_num(_pick(r, "High", "Highestprice")) or 0.0),
                low=float(_num(_pick(r, "Low", "Lowestprice")) or 0.0),
                close=float(_num(_pick(r, "Close", "Closeprice")) or 0.0),
                volume=float(_num(_pick(r, "Volume", "Totalmatchvol")) or 0.0),
                value=_num(_pick(r, "Value", "Totalmatchval")),
                data_source="ssi_fastconnect",
            )
        )
        meta.append(
            {
                "symbol": str(r["Symbol"]),
                "ts_utc": _parse_dt(r),
                "ref_price": _num(_pick(r, "Ref", "RefPrice")),
                "ceiling_price": _num(_pick(r, "Ceiling", "CeilingPrice")),
                "floor_price": _num(_pick(r, "Floor", "FloorPrice")),
            }
        )
    return bars, meta


def map_stream_ol(msg: dict[str, Any]) -> OddLotSnapshot:
    return OddLotSnapshot(
        symbol=str(msg.get("Symbol", "")),
        ts_utc=_parse_dt(msg),
        last_price=_num(msg.get("LastPrice")),
        last_vol=_num(msg.get("LastVol")),
        total_val=_num(msg.get("TotalVal")),
        total_vol=_num(msg.get("TotalVol")),
        bid_prices=[float(_num(msg.get(f"BidPrice{i}")) or 0.0) for i in range(1, 4)],
        bid_volumes=[float(_num(msg.get(f"BidVol{i}")) or 0.0) for i in range(1, 4)],
        ask_prices=[float(_num(msg.get(f"AskPrice{i}")) or 0.0) for i in range(1, 4)],
        ask_volumes=[float(_num(msg.get(f"AskVol{i}")) or 0.0) for i in range(1, 4)],
        trading_status=(
            str(_pick(msg, "TradingStatus")) if _pick(msg, "TradingStatus") is not None else None
        ),
        trading_session=(
            str(_pick(msg, "TradingSession")) if _pick(msg, "TradingSession") is not None else None
        ),
    )

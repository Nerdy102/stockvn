import json
from pathlib import Path

from data.providers.ssi_fastconnect.mappers import (
    map_stream_b,
    map_stream_mi,
    map_stream_ol,
    map_stream_r,
    map_stream_x_quote,
    map_stream_x_trade,
)

FIX = Path("tests/fixtures/ssi_fcdata")


def _content(name: str) -> dict:
    raw = json.loads((FIX / name).read_text())
    c = raw["Content"]
    return json.loads(c) if isinstance(c, str) else c


def test_x_quote_and_x_trade_mapping() -> None:
    q = map_stream_x_quote(_content("streaming_x_quote.json"))
    t = map_stream_x_trade(_content("streaming_x_trade.json"))
    assert q.symbol == "FPT"
    assert len(q.bid_prices) == 10
    assert t.last_price == 101


def test_r_mi_b_alias_tolerant() -> None:
    r = map_stream_r(
        {
            "Symbol": "FPT",
            "TradingDate": "10/01/2025",
            "Time": "10:03:00",
            "FBuyVol": "10",
            "FSellVol": "5",
        }
    )
    mi = map_stream_mi(
        {
            "IndexId": "VNINDEX",
            "TradingDate": "10/01/2025",
            "Time": "100300",
            "IndexValue": "1230",
            "PriorIndexValue": "1220",
        }
    )
    b = map_stream_b(
        {**_content("streaming_b.json"), "TradingDate": "10/01/2025", "TradingTime": "10:04:00"}
    )
    assert r.buy_vol == 10
    assert mi.index_id == "VNINDEX"
    assert b.timeframe == "1m"


def test_ol_mapping_three_levels() -> None:
    ol = map_stream_ol(
        {
            "Symbol": "FPT",
            "TradingDate": "10/01/2025",
            "Time": "10:05:00",
            "LastPrice": "100",
            "BidPrice1": "99",
            "BidPrice2": "98",
            "BidPrice3": "97",
            "AskPrice1": "101",
            "AskPrice2": "102",
            "AskPrice3": "103",
            "TradingStatus": "OPEN",
        }
    )
    assert len(ol.bid_prices) == 3
    assert ol.trading_status == "OPEN"

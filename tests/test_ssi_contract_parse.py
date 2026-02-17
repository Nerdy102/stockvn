from __future__ import annotations

import json
from pathlib import Path

from core.db.models import PriceOHLCV
from data.adapters.ssi_mapper import SSIMapper
from data.schemas.ssi_fcdata import (
    DailyIndexRecord,
    DailyOhlcRecord,
    DailyStockPriceRecord,
    SecuritiesDetailsResponse,
    SecurityRecord,
    StreamingB,
    StreamingEnvelope,
    StreamingMI,
    StreamingR,
    StreamingXQuote,
    StreamingXSnapshot,
    StreamingXTrade,
)
from sqlmodel import Session, SQLModel, create_engine, select

FIX = Path("tests/fixtures/ssi_fcdata")


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_contract_models_parse_and_aliases() -> None:
    sec = SecurityRecord.model_validate(_load("securities.json")[0])
    assert sec.Symbol == "FPT"

    details = SecuritiesDetailsResponse.model_validate(_load("securities_details.json"))
    assert details.RepeatedInfo[0].LotSize == 100

    ohlc = DailyOhlcRecord.model_validate(_load("daily_ohlc.json")[0])
    assert "NewField" in ohlc.unknown_fields()

    index = DailyIndexRecord.model_validate(_load("daily_index.json")[0])
    assert index.Indexcode == "VNINDEX"

    dsp = DailyStockPriceRecord.model_validate(_load("daily_stock_price.json")[0])
    assert dsp.Foreignsellvaltotal == 1200000000
    assert dsp.Netforeignvol == 10000


def test_streaming_envelope_parse_string_and_object_content() -> None:
    x_quote = StreamingEnvelope.model_validate(_load("streaming_x_quote.json"))
    assert StreamingXQuote.model_validate(x_quote.content_as_dict()).Symbol == "FPT"

    x_trade = StreamingEnvelope.model_validate(_load("streaming_x_trade.json"))
    assert StreamingXTrade.model_validate(x_trade.content_as_dict()).LastPrice == 101

    x_snap = StreamingEnvelope.model_validate(_load("streaming_x_snapshot.json"))
    assert StreamingXSnapshot.model_validate(x_snap.content_as_dict()).Symbol == "FPT"

    mi = StreamingEnvelope.model_validate(_load("streaming_mi.json"))
    assert StreamingMI.model_validate(mi.content_as_dict()).IndexId == "VNINDEX"

    r = StreamingEnvelope.model_validate(_load("streaming_r.json"))
    assert StreamingR.model_validate(r.content_as_dict()).CurrentRoom == 300000

    b = StreamingEnvelope.model_validate(_load("streaming_b.json"))
    assert StreamingB.model_validate(b.content_as_dict()).Symbol == "FPT"


def test_mapper_and_upsert_idempotency() -> None:
    record = _load("daily_ohlc.json")[0]
    bar = SSIMapper.map_daily_ohlc(record)

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        for _ in range(2):
            s.merge(
                PriceOHLCV(
                    symbol=bar.symbol,
                    timeframe=bar.timeframe,
                    timestamp=bar.ts,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    value_vnd=bar.value or 0.0,
                )
            )
            s.commit()

        rows = s.exec(select(PriceOHLCV)).all()
        assert len(rows) == 1

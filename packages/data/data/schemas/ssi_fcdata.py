from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.metrics import METRICS

log = logging.getLogger(__name__)

VN_TZ = dt.timezone(dt.timedelta(hours=7))


class SSIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def unknown_fields(self) -> list[str]:
        known = set(type(self).model_fields.keys())
        return sorted([k for k in self.model_extra.keys() if k not in known]) if self.model_extra else []

    def record_schema_drift_metrics(self, model_name: str | None = None) -> None:
        for field_name in self.unknown_fields():
            METRICS.inc("schema_unknown_fields_total", model=model_name or type(self).__name__, field_name=field_name)
            log.warning("schema_unknown_field", extra={"event": "schema_drift", "app_module": "ssi_schema", "field_name": field_name, "model": model_name or type(self).__name__})


# ---------- Coercion helpers ----------
def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    return float(s)


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


def _parse_date(v: Any) -> dt.date | None:
    if v in (None, ""):
        return None
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    s = str(v).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _parse_dt(v: Any, trading_date: dt.date | None = None) -> dt.datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=VN_TZ)
    s = str(v).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(s, fmt)
            return parsed.replace(tzinfo=VN_TZ)
        except ValueError:
            pass
    if trading_date:
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = dt.datetime.strptime(s, fmt).time()
                return dt.datetime.combine(trading_date, t, tzinfo=VN_TZ)
            except ValueError:
                pass
    return None


class SecurityRecord(SSIBaseModel):
    Market: str
    Symbol: str
    StockName: str | None = None
    StockEnName: str | None = None


class SecurityDetailsItem(SSIBaseModel):
    Isin: str | None = None
    Symbol: str
    SymbolName: str | None = None
    SymbolEngName: str | None = None
    SecType: str | None = None
    MarketId: str | None = None
    Exchange: str | None = None
    Issuer: str | None = None
    LotSize: int | None = None
    IssueDate: dt.date | None = None
    MaturityDate: dt.date | None = None
    FirstTradingDate: dt.date | None = None
    LastTradingDate: dt.date | None = None
    ContractMultiplier: float | None = None
    SettlMethod: str | None = None
    Underlying: str | None = None
    PutOrCall: str | None = None
    ExercisePrice: float | None = None
    ExerciseStyle: str | None = None
    ExcerciseRatio: str | None = None
    ListedShare: int | None = None
    TickPrice1: float | None = None
    TickPrice2: float | None = None
    TickPrice3: float | None = None
    TickPrice4: float | None = None
    TickIncrement1: float | None = None
    TickIncrement2: float | None = None
    TickIncrement3: float | None = None
    TickIncrement4: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for fld in ["LotSize", "ListedShare"]:
            data[fld] = _to_int(data.get(fld))
        for fld in [
            "ContractMultiplier",
            "ExercisePrice",
            "TickPrice1",
            "TickPrice2",
            "TickPrice3",
            "TickPrice4",
            "TickIncrement1",
            "TickIncrement2",
            "TickIncrement3",
            "TickIncrement4",
        ]:
            data[fld] = _to_float(data.get(fld))
        for fld in ["IssueDate", "MaturityDate", "FirstTradingDate", "LastTradingDate"]:
            data[fld] = _parse_date(data.get(fld))
        return data


class SecuritiesDetailsResponse(SSIBaseModel):
    RType: str | None = None
    ReportDate: dt.date | None = None
    TotalNoSym: int | None = None
    RepeatedInfo: list[SecurityDetailsItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["ReportDate"] = _parse_date(data.get("ReportDate"))
            data["TotalNoSym"] = _to_int(data.get("TotalNoSym"))
        return data


class DailyOhlcRecord(SSIBaseModel):
    Symbol: str
    Market: str
    TradingDate: dt.date
    Time: dt.datetime | None = None
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Value: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("TradingDate"))
        data["TradingDate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        for fld in ["Open", "High", "Low", "Close", "Volume", "Value"]:
            data[fld] = _to_float(data.get(fld))
        return data


class DailyIndexRecord(SSIBaseModel):
    Indexcode: str
    IndexValue: float | None = None
    TradingDate: dt.date
    Time: dt.datetime | None = None
    Change: float | None = None
    RatioChange: float | None = None
    TotalTrade: float | None = None
    Totalmatchvol: float | None = None
    Totalmatchval: float | None = None
    TypeIndex: str | None = None
    IndexName: str | None = None
    Advances: int | None = None
    Nochanges: int | None = None
    Declines: int | None = None
    Ceiling: int | None = None
    Floor: int | None = None
    TradingSession: str | None = None
    Market: str | None = None
    Exchange: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("TradingDate"))
        data["TradingDate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        for fld in ["IndexValue", "Change", "RatioChange", "TotalTrade", "Totalmatchvol", "Totalmatchval"]:
            data[fld] = _to_float(data.get(fld))
        return data


class DailyStockPriceRecord(SSIBaseModel):
    Tradingdate: dt.date
    Symbol: str
    Openprice: float | None = None
    Highestprice: float | None = None
    Lowestprice: float | None = None
    Closeprice: float | None = None
    Totalmatchvol: float | None = None
    Totalmatchval: float | None = None
    Foreignsellvaltotal: float | None = Field(default=None, validation_alias="Toreignsellvaltotal")
    Netforeignvol: float | None = Field(default=None, validation_alias="Netforeivol")
    Time: dt.datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("Tradingdate"))
        data["Tradingdate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        for fld in [
            "Openprice",
            "Highestprice",
            "Lowestprice",
            "Closeprice",
            "Totalmatchvol",
            "Totalmatchval",
            "Toreignsellvaltotal",
            "Netforeivol",
        ]:
            data[fld] = _to_float(data.get(fld))
        return data


class IndexListRecord(SSIBaseModel):
    IndexCode: str
    IndexName: str | None = None
    Exchange: str | None = None


class IndexComponentItem(SSIBaseModel):
    Isin: str | None = None
    StockSymbol: str | None = None


class IndexComponentsResponse(SSIBaseModel):
    IndexCode: str
    IndexName: str | None = None
    Exchange: str | None = None
    TotalSymbolNo: int | None = None
    IndexComponent: list[IndexComponentItem] = Field(default_factory=list)


class StreamingF(SSIBaseModel):
    RType: Literal["F"]
    Symbol: str
    TradingDate: dt.date | None = None
    Time: dt.datetime | None = None
    TradingSession: str | None = None
    TradingStatus: str | None = None
    Exchange: str | None = None


class StreamingEnvelope(SSIBaseModel):
    DataType: str
    Content: dict[str, Any] | str

    def content_as_dict(self) -> dict[str, Any]:
        if isinstance(self.Content, dict):
            return self.Content
        return json.loads(self.Content)


class StreamingXTrade(SSIBaseModel):
    RType: Literal["X-TRADE"]
    Symbol: str
    TradingDate: dt.date | None = None
    Time: dt.datetime | None = None
    LastPrice: float | None = None
    LastVol: float | None = None
    TotalVal: float | None = None
    TotalVol: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("TradingDate"))
        data["TradingDate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        for fld in ["LastPrice", "LastVol", "TotalVal", "TotalVol"]:
            data[fld] = _to_float(data.get(fld))
        return data


class StreamingXQuote(SSIBaseModel):
    RType: Literal["X-QUOTE"]
    Symbol: str
    TradingDate: dt.date | None = None
    Time: dt.datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("TradingDate"))
        data["TradingDate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        return data


class StreamingXSnapshot(SSIBaseModel):
    RType: Literal["X"]
    Symbol: str
    TradingDate: dt.date | None = None
    Time: dt.datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        td = _parse_date(data.get("TradingDate"))
        data["TradingDate"] = td
        data["Time"] = _parse_dt(data.get("Time"), td)
        return data


class StreamingMI(SSIBaseModel):
    RType: Literal["MI"]
    IndexId: str
    Time: dt.datetime | None = None
    IndexValue: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        data["Time"] = _parse_dt(data.get("Time"))
        data["IndexValue"] = _to_float(data.get("IndexValue"))
        return data


class StreamingR(SSIBaseModel):
    RType: Literal["R"]
    Symbol: str
    TotalRoom: float | None = None
    CurrentRoom: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        data["TotalRoom"] = _to_float(data.get("TotalRoom"))
        data["CurrentRoom"] = _to_float(data.get("CurrentRoom"))
        return data


class StreamingB(SSIBaseModel):
    RType: Literal["B"]
    Symbol: str
    Time: dt.datetime | None = None
    Open: float | None = None
    High: float | None = None
    Low: float | None = None
    Close: float | None = None
    Volume: float | None = None
    Value: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        data["Time"] = _parse_dt(data.get("Time"))
        for fld in ["Open", "High", "Low", "Close", "Volume", "Value"]:
            data[fld] = _to_float(data.get(fld))
        return data

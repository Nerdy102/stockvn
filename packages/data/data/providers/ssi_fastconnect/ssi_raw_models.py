from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ALIAS = {
    "MarketID": "market_id",
    "MarketId": "market_id",
    "Exchange": "exchange",
    "Isin": "isin",
    "ISIN": "isin",
    "TotalMatchVol": "total_match_vol",
    "Totalmatchvol": "total_match_vol",
    "TotalMatchVOL": "total_match_vol",
    "TotalMatchVal": "total_match_val",
    "Totalmatchval": "total_match_val",
    "ForeignSellValTotal": "foreign_sell_val_total",
    "Toreignsellvaltotal": "foreign_sell_val_total",
    "NetForeiVol": "net_foreign_vol",
    "Netforeivol": "net_foreign_vol",
    "Tradingdate": "trading_date",
    "TradingDate": "trading_date",
}


class SsiRawMessage(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    datatype: str | None = None
    content: dict[str, Any] | str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _parse_content(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
        return v

    def normalized_content(self) -> dict[str, Any]:
        payload = self.content if isinstance(self.content, dict) else {}
        out: dict[str, Any] = {}
        for k, v in payload.items():
            out[_ALIAS.get(k, k)] = v
        return out


class SsiRawEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: int | str | None = None
    message: str | None = None
    data: list[dict[str, Any]] | dict[str, Any] | None = None
    access_token: str | None = Field(default=None, alias="accessToken")

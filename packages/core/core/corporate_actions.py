from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

WITHHOLDING_TAX_RATE = 0.05


@dataclass(frozen=True)
class CorporateActionEvent:
    symbol: str
    action_type: str
    ex_date: dt.date
    params: dict[str, Any]
    record_date: dt.date | None = None
    pay_date: dt.date | None = None
    public_date: dt.date | None = None


def _to_date(v: Any) -> dt.date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, dt.date):
        return v
    return pd.to_datetime(v).date()


def normalize_corporate_actions(ca: pd.DataFrame | list[dict[str, Any]] | None) -> list[CorporateActionEvent]:
    if ca is None:
        return []
    rows = ca if isinstance(ca, list) else ca.to_dict(orient="records")
    out: list[CorporateActionEvent] = []
    for row in rows:
        params = dict(row.get("params_json") or row.get("params") or {})
        out.append(
            CorporateActionEvent(
                symbol=str(row["symbol"]),
                action_type=str(row["action_type"]).upper(),
                ex_date=_to_date(row.get("ex_date")) or dt.date.min,
                record_date=_to_date(row.get("record_date")),
                pay_date=_to_date(row.get("pay_date")),
                public_date=_to_date(row.get("public_date")),
                params=params,
            )
        )
    return out


def effective_corporate_actions(
    actions: list[CorporateActionEvent],
    symbol: str,
    as_of_date: dt.date | None,
) -> list[CorporateActionEvent]:
    out: list[CorporateActionEvent] = []
    for event in actions:
        if event.symbol != symbol:
            continue
        if as_of_date is not None and event.public_date is not None and event.public_date > as_of_date:
            continue
        out.append(event)
    return sorted(out, key=lambda x: (x.ex_date, x.action_type))


def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.set_index("date", drop=False)
    elif "timestamp" in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"]).dt.date
        out = out.set_index("date", drop=False)
    else:
        out = out.copy()
        out.index = pd.to_datetime(out.index).date
        out["date"] = out.index
    return out.sort_index()


def compute_rights_adjustment(prev_close: float, ratio: float, sub_price: float) -> tuple[float, float]:
    terp = (float(prev_close) + float(ratio) * float(sub_price)) / (1.0 + float(ratio))
    right_value = max(float(prev_close) - float(terp), 0.0)
    return float(terp), float(right_value)


def apply_price_adjustments(
    bars: pd.DataFrame,
    actions: list[CorporateActionEvent],
    *,
    adjusted: bool,
    total_return: bool,
) -> pd.DataFrame:
    out = _ensure_date_index(bars)
    if "close" not in out.columns:
        raise ValueError("Missing OHLCV column: close")
    for col in ["open", "high", "low"]:
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out["adj_factor_price"] = 1.0
    out["adj_factor_volume"] = 1.0
    out["tr_index"] = 1.0

    split_events = [a for a in actions if a.action_type == "SPLIT"]
    rights_events = [a for a in actions if a.action_type == "RIGHTS_ISSUE"]
    div_events = [a for a in actions if a.action_type == "CASH_DIVIDEND"]

    # Backward adjustment using multiplicative factor before ex_date, preserving all dates.
    for ev in split_events:
        factor = float(ev.params.get("split_factor", 1.0))
        if factor <= 0:
            continue
        mask = out.index < ev.ex_date
        out.loc[mask, "adj_factor_price"] /= factor
        out.loc[mask, "adj_factor_volume"] *= factor

    for ev in rights_events:
        ratio = float(ev.params.get("ratio", 0.0))
        sub_price = float(ev.params.get("subscription_price", 0.0))
        prev_rows = out[out.index < ev.ex_date]
        if prev_rows.empty:
            continue
        prev_close = float(prev_rows.iloc[-1]["close"])
        terp, _ = compute_rights_adjustment(prev_close, ratio, sub_price)
        if prev_close <= 0:
            continue
        factor = terp / prev_close
        mask = out.index < ev.ex_date
        out.loc[mask, "adj_factor_price"] *= factor

    if adjusted:
        for c in ["open", "high", "low", "close"]:
            out[c] = out[c] * out["adj_factor_price"]
        out["volume"] = out["volume"] * out["adj_factor_volume"]
        out["is_adjusted"] = True
    else:
        out["is_adjusted"] = False

    if total_return:
        idx = 1.0
        closes = out["close"].astype(float)
        by_date_div: dict[dt.date, float] = {}
        for ev in div_events:
            cash_per_share = float(ev.params.get("cash_per_share", 0.0))
            credit_dt = ev.pay_date or ev.ex_date
            by_date_div[credit_dt] = by_date_div.get(credit_dt, 0.0) + cash_per_share

        tr_values: list[float] = []
        prev_close: float | None = None
        for d, px in closes.items():
            if prev_close is None:
                tr_values.append(idx)
                prev_close = float(px)
                continue
            div = by_date_div.get(d, 0.0)
            idx *= (float(px) + float(div)) / float(prev_close)
            tr_values.append(float(idx))
            prev_close = float(px)
        out["tr_index"] = tr_values

    return out.reset_index(drop=True)


def apply_ca_to_position(
    *,
    qty_before: float,
    avg_cost_before: float,
    ex_date_price: float,
    action: CorporateActionEvent,
    fee_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    credit_dividend_on_ex_date: bool = False,
) -> dict[str, Any]:
    action_type = action.action_type
    qty_after = float(qty_before)
    avg_cost_after = float(avg_cost_before)
    cash_delta = 0.0
    fee = 0.0
    tax = 0.0
    notes: dict[str, Any] = {}
    cash_posting_date = action.pay_date or action.ex_date

    if action_type == "SPLIT":
        factor = float(action.params.get("split_factor", 1.0))
        qty_after = qty_before * factor
        avg_cost_after = avg_cost_before / factor if factor > 0 else avg_cost_before
    elif action_type == "CASH_DIVIDEND":
        if action.pay_date is None and not credit_dividend_on_ex_date:
            cash_posting_date = action.ex_date
        elif action.pay_date is not None:
            cash_posting_date = action.pay_date
        gross = qty_before * float(action.params.get("cash_per_share", 0.0))
        tax = gross * WITHHOLDING_TAX_RATE
        cash_delta = gross - tax
    elif action_type == "RIGHTS_ISSUE":
        ratio = float(action.params.get("ratio", 0.0))
        sub_price = float(action.params.get("subscription_price", 0.0))
        terp, right_value = compute_rights_adjustment(ex_date_price, ratio, sub_price)
        notional = qty_before * right_value
        fee = notional * fee_rate
        tax = notional * sell_tax_rate
        cash_delta = notional - fee - tax
        notes = {"terp": terp, "right_value": right_value, "notional": notional, "policy": "sell_rights"}
    else:
        raise ValueError(f"Unsupported corporate action: {action_type}")

    return {
        "symbol": action.symbol,
        "action_type": action.action_type,
        "ex_date": action.ex_date,
        "qty_before": float(qty_before),
        "qty_after": float(qty_after),
        "cash_delta": float(cash_delta),
        "avg_cost_before": float(avg_cost_before),
        "avg_cost_after": float(avg_cost_after),
        "fee": float(fee),
        "tax": float(tax),
        "cash_posting_date": cash_posting_date,
        "notes_json": notes,
    }


def adjust_prices(
    symbol: str,
    bars: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    method: str = "none",
    *,
    corporate_actions: pd.DataFrame | list[dict[str, Any]] | None = None,
    as_of_date: dt.date | None = None,
    total_return: bool = False,
) -> pd.DataFrame:
    out = bars.copy()
    if method == "none" and corporate_actions is None:
        out["is_adjusted"] = False
        return out

    actions = effective_corporate_actions(
        normalize_corporate_actions(corporate_actions),
        symbol=symbol,
        as_of_date=as_of_date,
    )
    adjusted = method != "none"
    return apply_price_adjustments(out, actions, adjusted=adjusted, total_return=total_return)

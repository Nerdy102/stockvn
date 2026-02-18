from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pandas as pd
from core.db.models import (
    AlphaPrediction,
    AnnotationAudit,
    CorporateAction,
    PriceOHLCV,
    UserAnnotationV2,
)
from core.indicators import add_indicators
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/chart", tags=["chart"])

_ALLOWED_SHAPE_FIELDS = {"type", "x0", "x1", "y0", "y1", "line", "fillcolor", "opacity"}
_ALLOWED_LINE_FIELDS = {"color", "width", "dash"}
_ALLOWED_SHAPE_TYPES = {"line", "rect"}


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid date: {value}") from exc


def _parse_dt(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        try:
            return dt.datetime.fromisoformat(value + "T00:00:00")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid datetime: {value}") from exc


def _resample_weekly(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv
    w = (
        ohlcv.set_index("timestamp")
        .resample("W-FRI")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "value_vnd": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return w


def _apply_adjustment(rows: list[PriceOHLCV], actions: list[CorporateAction]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.model_dump() for r in rows]).sort_values("timestamp")
    if not actions:
        return df
    for ca in actions:
        if ca.action_type.lower() == "split":
            factor = float((ca.params_json or {}).get("split_factor", 1.0) or 1.0)
            if factor > 0:
                mask = pd.to_datetime(df["timestamp"]).dt.date < ca.ex_date
                df.loc[mask, ["open", "high", "low", "close"]] = (
                    df.loc[mask, ["open", "high", "low", "close"]] / factor
                )
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * factor
        if ca.action_type.lower() == "cash_dividend":
            cash = float((ca.params_json or {}).get("cash", 0.0) or 0.0)
            mask = pd.to_datetime(df["timestamp"]).dt.date < ca.ex_date
            df.loc[mask, ["open", "high", "low", "close"]] = (
                df.loc[mask, ["open", "high", "low", "close"]] - cash
            ).clip(lower=0.0)
    return df


@router.get("/ohlcv")
def get_chart_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    adjusted: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)

    tf = "1D" if timeframe == "1W" else timeframe
    rows = list(
        db.exec(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.timeframe == tf)
            .where(PriceOHLCV.timestamp >= start_dt)
            .where(PriceOHLCV.timestamp <= end_dt)
            .order_by(PriceOHLCV.timestamp)
        ).all()
    )

    ca_rows = list(
        db.exec(
            select(CorporateAction)
            .where(CorporateAction.symbol == symbol)
            .where(CorporateAction.ex_date >= start_dt.date())
            .where(CorporateAction.ex_date <= end_dt.date())
            .order_by(CorporateAction.ex_date)
        ).all()
    )

    if adjusted:
        df = _apply_adjustment(rows, ca_rows)
    else:
        df = pd.DataFrame([r.model_dump() for r in rows]) if rows else pd.DataFrame()

    if df.empty:
        return {"rows": [], "ca_markers": []}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if timeframe == "1W":
        df = _resample_weekly(df)

    return {
        "rows": df[["timestamp", "open", "high", "low", "close", "volume", "value_vnd"]]
        .assign(timestamp=lambda x: x["timestamp"].astype(str))
        .to_dict(orient="records"),
        "ca_markers": [
            {
                "ex_date": r.ex_date.isoformat(),
                "action_type": r.action_type,
                "params": r.params_json,
            }
            for r in ca_rows
        ],
    }


@router.get("/indicators")
def get_chart_indicators(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    adjusted: bool = Query(default=True),
    indicators: str = Query(default="SMA20,EMA20,RSI14,MACD,ATR14,VWAP"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    base = get_chart_ohlcv(
        symbol=symbol, timeframe=timeframe, start=start, end=end, adjusted=adjusted, db=db
    )
    rows = base.get("rows", [])
    if not rows:
        return {"rows": []}

    o = pd.DataFrame(rows)
    o["timestamp"] = pd.to_datetime(o["timestamp"])
    o = o.sort_values("timestamp").set_index("timestamp")
    ind = add_indicators(o)

    selected = [x.strip().upper() for x in indicators.split(",") if x.strip()]
    mapping = {
        "SMA20": "SMA20",
        "EMA20": "EMA20",
        "RSI14": "RSI14",
        "MACD": "MACD",
        "ATR14": "ATR14",
        "VWAP": "VWAP",
        "MACD_SIGNAL": "MACD_SIGNAL",
        "MACD_HIST": "MACD_HIST",
    }
    keep_cols = [mapping[x] for x in selected if x in mapping and mapping[x] in ind.columns]
    if "MACD" in selected:
        for extra in ["MACD_SIGNAL", "MACD_HIST"]:
            if extra in ind.columns and extra not in keep_cols:
                keep_cols.append(extra)
    out = ind.reset_index()[["timestamp"] + keep_cols]
    out["timestamp"] = out["timestamp"].astype(str)
    return {"rows": out.to_dict(orient="records")}


@router.get("/alpha")
def get_chart_alpha(
    symbol: str,
    start: str,
    end: str,
    model_id: str = Query(default="alpha_v3"),
    timeframe: str = Query(default="1D"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    start_d = _parse_date(start)
    end_d = _parse_date(end)
    rows = list(
        db.exec(
            select(AlphaPrediction)
            .where(AlphaPrediction.symbol == symbol)
            .where(AlphaPrediction.model_id == model_id)
            .where(AlphaPrediction.as_of_date >= start_d)
            .where(AlphaPrediction.as_of_date <= end_d)
            .order_by(AlphaPrediction.as_of_date)
        ).all()
    )
    if not rows:
        return {"rows": []}
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["date"] = pd.to_datetime(df["as_of_date"])
    df["lo"] = df["mu"] - df["uncert"]
    df["hi"] = df["mu"] + df["uncert"]
    df["mu_norm"] = (df["mu"] - df["mu"].mean()) / (
        df["mu"].std(ddof=0) if df["mu"].std(ddof=0) else 1.0
    )
    df["lo_norm"] = (df["lo"] - df["mu"].mean()) / (
        df["mu"].std(ddof=0) if df["mu"].std(ddof=0) else 1.0
    )
    df["hi_norm"] = (df["hi"] - df["mu"].mean()) / (
        df["mu"].std(ddof=0) if df["mu"].std(ddof=0) else 1.0
    )

    if timeframe in {"15m", "60m"}:
        full = pd.date_range(start=pd.to_datetime(start_d), end=pd.to_datetime(end_d), freq="D")
        dff = pd.DataFrame({"date": full}).merge(
            df[["date", "mu_norm", "lo_norm", "hi_norm"]], on="date", how="left"
        )
        dff[["mu_norm", "lo_norm", "hi_norm"]] = dff[["mu_norm", "lo_norm", "hi_norm"]].ffill()
        dff["date"] = dff["date"].astype(str)
        return {"rows": dff.to_dict(orient="records")}

    df["date"] = df["date"].astype(str)
    return {"rows": df[["date", "mu_norm", "lo_norm", "hi_norm"]].to_dict(orient="records")}


class AnnotationPost(BaseModel):
    workspace_id: str
    symbol: str
    timeframe: str
    window_start: dt.date
    window_end: dt.date
    actor: str = "user"
    notes: str = ""
    shapes_json: list[dict[str, Any]] = Field(default_factory=list)


def validate_shapes_json(shapes: list[dict[str, Any]]) -> None:
    for shape in shapes:
        unknown = set(shape.keys()) - _ALLOWED_SHAPE_FIELDS
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown shape fields: {sorted(unknown)}")
        s_type = shape.get("type")
        if s_type not in _ALLOWED_SHAPE_TYPES:
            raise HTTPException(status_code=400, detail="shape type must be line|rect")
        if "line" in shape:
            line = shape["line"]
            if not isinstance(line, dict):
                raise HTTPException(status_code=400, detail="line must be object")
            unknown_line = set(line.keys()) - _ALLOWED_LINE_FIELDS
            if unknown_line:
                raise HTTPException(
                    status_code=400, detail=f"unknown line fields: {sorted(unknown_line)}"
                )


@router.get("/annotations")
def get_annotations(
    workspace_id: str,
    symbol: str,
    timeframe: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = list(
        db.exec(
            select(UserAnnotationV2)
            .where(UserAnnotationV2.workspace_id == workspace_id)
            .where(UserAnnotationV2.symbol == symbol)
            .where(UserAnnotationV2.timeframe == timeframe)
            .order_by(UserAnnotationV2.version.desc())
        ).all()
    )
    return {
        "versions": [r.model_dump() for r in rows],
        "latest": rows[0].model_dump() if rows else None,
    }


@router.post("/annotations")
def save_annotations(payload: AnnotationPost, db: Session = Depends(get_db)) -> dict[str, Any]:
    validate_shapes_json(payload.shapes_json)

    existing = list(
        db.exec(
            select(UserAnnotationV2)
            .where(UserAnnotationV2.workspace_id == payload.workspace_id)
            .where(UserAnnotationV2.symbol == payload.symbol)
            .where(UserAnnotationV2.timeframe == payload.timeframe)
            .order_by(UserAnnotationV2.version.desc())
        ).all()
    )
    next_version = (existing[0].version + 1) if existing else 1

    row = UserAnnotationV2(
        id=str(uuid.uuid4()),
        workspace_id=payload.workspace_id,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.window_start,
        end_date=payload.window_end,
        version=next_version,
        shapes_json={"shapes": payload.shapes_json},
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    db.add(row)
    db.flush()

    db.add(
        AnnotationAudit(
            annotation_id=row.id,
            action="SAVE",
            action_at=dt.datetime.utcnow(),
            actor=payload.actor,
            notes=payload.notes,
        )
    )

    rows = existing
    if len(rows) >= 10:
        to_delete = rows[9:]
        for old in to_delete:
            db.delete(old)

    db.commit()
    db.refresh(row)
    return {"id": row.id, "version": row.version, "shapes_json": row.shapes_json}

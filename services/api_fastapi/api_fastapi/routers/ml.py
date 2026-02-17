from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd
from core.db.models import BacktestRun, MlPrediction, PriceOHLCV
from core.ml.backtest import run_sensitivity, run_stress, run_walk_forward
from core.ml.features import build_ml_features, feature_columns
from core.ml.models import MlModelBundle
from core.ml.reports import build_metrics_table, disclaimer
from core.settings import Settings, get_settings
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/ml", tags=["ml"])
_MODELS = ["ridge_v1", "hgbr_v1", "ensemble_v1"]


@router.get("/models")
def get_models() -> list[str]:
    return _MODELS


def _load_features(db: Session) -> pd.DataFrame:
    rows = list(db.exec(select(PriceOHLCV)).all())
    if not rows:
        return pd.DataFrame()
    return build_ml_features(pd.DataFrame([r.model_dump() for r in rows]))


@router.post("/train")
def train_models(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")
    feat = _load_features(db)
    feat = feat.dropna(subset=["y_excess"])
    if feat.empty:
        return {"status": "insufficient_data"}

    cols = feature_columns(feat)
    model = MlModelBundle().fit(feat[cols], feat["y_excess"])
    y_hat = model.predict(feat[cols])

    for j, (_, row) in enumerate(feat.iterrows()):
        old = db.exec(
            select(MlPrediction)
            .where(MlPrediction.model_id == "ensemble_v1")
            .where(MlPrediction.symbol == str(row["symbol"]))
            .where(MlPrediction.as_of_date == row["as_of_date"])
        ).first()
        if old:
            old.y_hat = float(y_hat[j])
            db.add(old)
        else:
            db.add(
                MlPrediction(
                    model_id="ensemble_v1",
                    symbol=str(row["symbol"]),
                    as_of_date=row["as_of_date"],
                    y_hat=float(y_hat[j]),
                    meta={"feature_version": "v1"},
                )
            )
    db.commit()
    return {"status": "ok", "rows": int(len(feat)), "models": _MODELS}


@router.get("/predict")
def predict(
    date: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    universe: str = Query(default="ALL", pattern="^(ALL|VN30|VNINDEX)$"),
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    if not date and not (start and end):
        raise HTTPException(status_code=400, detail="date or start/end required")

    q = select(MlPrediction).where(MlPrediction.model_id == "ensemble_v1")
    if date:
        as_of = dt.datetime.strptime(date, "%d-%m-%Y").date()
        q = q.where(MlPrediction.as_of_date == as_of)
    else:
        s = dt.datetime.strptime(start, "%d-%m-%Y").date()
        e = dt.datetime.strptime(end, "%d-%m-%Y").date()
        if (e - s).days > 365:
            raise HTTPException(status_code=400, detail="max range 365 days without pagination")
        q = q.where(MlPrediction.as_of_date >= s).where(MlPrediction.as_of_date <= e)
    rows = list(db.exec(q.offset(offset).limit(limit)).all())
    return [r.model_dump() for r in rows]


@router.post("/backtest")
def backtest(payload: dict | None = None, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")

    payload = payload or {}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    old = db.exec(select(BacktestRun).where(BacktestRun.run_hash == key)).first()
    if old:
        return {"cached": True, **old.summary_json}

    feat = _load_features(db).dropna(subset=["y_excess"])
    if feat.empty:
        out = {
            "walk_forward": {"metrics": build_metrics_table({})},
            "sensitivity": run_sensitivity({}),
            "stress": run_stress({}),
            "disclaimer": disclaimer(),
        }
    else:
        cols = feature_columns(feat)
        curve, metrics = run_walk_forward(feat, cols)
        out = {
            "walk_forward": {
                "equity_curve": curve.to_dict(orient="records"),
                "metrics": build_metrics_table(metrics),
            },
            "sensitivity": run_sensitivity(metrics),
            "stress": run_stress(metrics),
            "disclaimer": disclaimer(),
        }

    db.add(BacktestRun(run_hash=key, config_json=payload, summary_json=out))
    db.commit()
    return {"cached": False, **out}

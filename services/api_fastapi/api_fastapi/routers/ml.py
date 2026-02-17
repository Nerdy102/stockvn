from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

import pandas as pd
from core.db.models import (
    BacktestRun,
    DiagnosticsMetric,
    DiagnosticsRun,
    MlPrediction,
    PriceOHLCV,
)
from core.ml.backtest import run_sensitivity, run_stress, run_walk_forward
from core.ml.diagnostics import (
    block_bootstrap_ci,
    capacity_proxy,
    decile_spread,
    ic_decay,
    rank_ic,
    regime_breakdown,
    turnover_cost_attribution,
)
from core.ml.features import build_ml_features, feature_columns
from core.ml.models import MlModelBundle
from core.ml.models_v2 import MlModelV2Bundle
from core.ml.portfolio_v2 import (
    apply_constraints_ordered,
    apply_no_trade_band,
    build_weights_ivp_uncertainty,
)
from core.ml.reports import build_metrics_table, disclaimer
from core.ml.targets import compute_rank_z_label
from core.settings import Settings, get_settings
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/ml", tags=["ml"])
_MODELS = [
    "ridge_v1",
    "hgbr_v1",
    "ensemble_v1",
    "ridge_rank_v2",
    "hgbr_rank_v2",
    "hgbr_q10_v2",
    "hgbr_q50_v2",
    "hgbr_q90_v2",
    "ensemble_v2",
]


@router.get("/models")
def get_models() -> list[str]:
    return _MODELS


def _load_features(db: Session) -> pd.DataFrame:
    rows = list(db.exec(select(PriceOHLCV)).all())
    if not rows:
        return pd.DataFrame()
    feat = build_ml_features(pd.DataFrame([r.model_dump() for r in rows]))
    feat = compute_rank_z_label(feat, col="y_excess")
    return feat


@router.post("/train")
def train_models(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")
    feat = _load_features(db).dropna(subset=["y_excess", "y_rank_z"])
    if feat.empty:
        return {"status": "insufficient_data"}

    cols = feature_columns(feat)
    v1 = MlModelBundle().fit(feat[cols], feat["y_excess"])
    y_hat_v1 = v1.predict(feat[cols])

    v2 = MlModelV2Bundle().fit(feat[cols], feat["y_rank_z"])
    comp = v2.predict_components(feat[cols])

    for j, (_, row) in enumerate(feat.iterrows()):
        sym = str(row["symbol"])
        d = row["as_of_date"]
        old_v1 = db.exec(
            select(MlPrediction)
            .where(MlPrediction.model_id == "ensemble_v1")
            .where(MlPrediction.symbol == sym)
            .where(MlPrediction.as_of_date == d)
        ).first()
        if old_v1:
            old_v1.y_hat = float(y_hat_v1[j])
            old_v1.meta = {"feature_version": "v1"}
            db.add(old_v1)
        else:
            db.add(
                MlPrediction(
                    model_id="ensemble_v1",
                    symbol=sym,
                    as_of_date=d,
                    y_hat=float(y_hat_v1[j]),
                    meta={"feature_version": "v1"},
                )
            )

        meta_v2 = {
            "score_final": float(comp["score_final"][j]),
            "mu": float(comp["mu"][j]),
            "uncert": float(comp["uncert"][j]),
            "score_rank_z": float(comp["score_rank_z"][j]),
        }
        old_v2 = db.exec(
            select(MlPrediction)
            .where(MlPrediction.model_id == "ensemble_v2")
            .where(MlPrediction.symbol == sym)
            .where(MlPrediction.as_of_date == d)
        ).first()
        if old_v2:
            old_v2.y_hat = float(comp["score_final"][j])
            old_v2.meta = meta_v2
            db.add(old_v2)
        else:
            db.add(
                MlPrediction(
                    model_id="ensemble_v2",
                    symbol=sym,
                    as_of_date=d,
                    y_hat=float(comp["score_final"][j]),
                    meta=meta_v2,
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

    q = select(MlPrediction).where(MlPrediction.model_id == "ensemble_v2")
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
    out = []
    for r in rows:
        out.append(
            {
                "symbol": r.symbol,
                "date": r.as_of_date.isoformat(),
                "score_final": float(r.meta.get("score_final", r.y_hat)),
                "mu": float(r.meta.get("mu", r.y_hat)),
                "uncert": float(r.meta.get("uncert", 0.0)),
                "score_rank_z": float(r.meta.get("score_rank_z", r.y_hat)),
            }
        )
    return out


@router.post("/diagnostics")
def diagnostics(payload: dict | None = None, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")
    feat = _load_features(db).dropna(subset=["y_excess"])
    if feat.empty:
        return {"status": "insufficient_data"}

    preds = pd.DataFrame(
        [p.model_dump() for p in db.exec(select(MlPrediction).where(MlPrediction.model_id == "ensemble_v2")).all()]
    )
    if preds.empty:
        return {"status": "missing_predictions"}
    preds["score_final"] = preds["meta"].apply(lambda m: float((m or {}).get("score_final", 0.0)) if isinstance(m, dict) else 0.0)
    df = feat.merge(preds[["symbol", "as_of_date", "score_final"]], on=["symbol", "as_of_date"], how="inner")
    df["net_ret"] = df["y_excess"].fillna(0.0)

    ic = rank_ic(df)
    metrics = {
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else 0.0,
        **ic_decay(df),
        "decile_spread": decile_spread(df),
        **turnover_cost_attribution(df),
        **capacity_proxy(df.assign(order_notional=1e7, adv20_value=df.get("adv20_value", 1e9), liq_bound=False)),
        **regime_breakdown(df.assign(regime="sideways")),
        **block_bootstrap_ci(df["net_ret"]),
    }

    run_id = str(uuid.uuid4())
    cfg_hash = hashlib.sha256(json.dumps(payload or {}, sort_keys=True).encode()).hexdigest()
    db.add(DiagnosticsRun(run_id=run_id, model_id="ensemble_v2", config_hash=cfg_hash))
    for k, v in metrics.items():
        db.add(DiagnosticsMetric(run_id=run_id, metric_name=k, metric_value=float(v)))
    db.commit()
    return {"run_id": run_id, "metrics": metrics}


@router.post("/backtest")
def backtest(payload: dict | None = None, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")

    payload = payload or {}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    old = db.exec(select(BacktestRun).where(BacktestRun.run_hash == key)).first()
    if old:
        return {"cached": True, **old.summary_json}

    feat = _load_features(db).dropna(subset=["y_excess", "y_rank_z"])
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
        p = feat[["symbol", "as_of_date", "vol_60d", "adv20_value", "sector"]].copy()
        p["score_final"] = feat["y_rank_z"]
        p["uncert"] = 0.1
        w = build_weights_ivp_uncertainty(p)
        w = apply_constraints_ordered(w, nav=1_000_000_000.0, risk_off=False)
        w["current_w"] = 0.0
        w["target_qty"] = 100
        w["order_notional"] = w["w"] * 1_000_000_000.0
        w = apply_no_trade_band(w)
        metrics["selected_names"] = float(len(w))
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

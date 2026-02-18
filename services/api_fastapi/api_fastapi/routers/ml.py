from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

import numpy as np
import pandas as pd
from core.alpha_v3.features import enforce_no_leakage_guard
from core.db.models import (
    BacktestRun,
    DiagnosticsMetric,
    DiagnosticsRun,
    ForeignRoom,
    MlPrediction,
    PriceOHLCV,
    QuoteL2,
    Ticker,
)
from core.market_rules import clamp_qty_to_board_lot
from core.ml.backtest import run_sensitivity, run_stress, run_walk_forward
from core.ml.diagnostics import run_diagnostics
from core.ml.features import build_ml_features, feature_columns
from core.ml.features_v2 import build_features_v2
from core.ml.models import MlModelBundle
from core.ml.models_v2 import MlModelV2Bundle
from core.ml.portfolio_v2 import (
    apply_constraints_ordered,
    apply_exposure_overlay,
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
MODEL_IDS = [
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


def _base_features(db: Session) -> pd.DataFrame:
    prices = list(db.exec(select(PriceOHLCV)).all())
    if not prices:
        return pd.DataFrame()
    feat = build_ml_features(pd.DataFrame([p.model_dump() for p in prices]))
    feat = compute_rank_z_label(feat, col="y_excess")
    return feat


def _regime_from_vnindex(features: pd.DataFrame) -> pd.DataFrame:
    vn = features[features["symbol"] == "VNINDEX"].copy()
    if vn.empty:
        return pd.DataFrame(columns=["as_of_date", "regime"])
    cond_trend = (
        (vn["ema20"] > vn["ema50"])
        & (vn["close"] > vn["ema50"])
        & ((vn["ema50"] - vn["ema50"].shift(10)) > 0)
    )
    cond_off = (vn["close"] < vn["ema50"]) & (vn["ema20"] < vn["ema50"])
    vn["regime"] = "sideways"
    vn.loc[cond_trend, "regime"] = "trend_up"
    vn.loc[cond_off, "regime"] = "risk_off"
    return vn[["as_of_date", "regime"]]


def _v2_features(db: Session) -> pd.DataFrame:
    base = _base_features(db)
    if base.empty:
        return base

    regime = _regime_from_vnindex(base)

    fr = pd.DataFrame([r.model_dump() for r in db.exec(select(ForeignRoom)).all()])
    foreign = pd.DataFrame()
    if not fr.empty:
        fr["as_of_date"] = pd.to_datetime(fr["timestamp"]).dt.date
        foreign = fr.groupby(["symbol", "as_of_date"], as_index=False).last()
        foreign["net_foreign_val"] = foreign["fbuy_val"].fillna(0.0) - foreign["fsell_val"].fillna(
            0.0
        )

    quotes = pd.DataFrame([r.model_dump() for r in db.exec(select(QuoteL2)).all()])
    intraday = pd.DataFrame(
        [
            r.model_dump()
            for r in db.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1m")).all()
        ]
    )

    return build_features_v2(base, regime, foreign, quotes, intraday)


def _top30_universe_symbols(db: Session, universe: str) -> set[str]:
    if universe == "VNINDEX":
        return {"VNINDEX"}
    tickers = list(db.exec(select(Ticker)).all())
    if not tickers:
        return set()
    if universe == "VN30":
        eligible = [
            t for t in tickers if t.exchange in {"HOSE", "HNX", "UPCOM"} and t.symbol != "VNINDEX"
        ]
        eligible = sorted(
            eligible, key=lambda t: float(getattr(t, "market_cap", 0.0) or 0.0), reverse=True
        )
        return {t.symbol for t in eligible[:30]}
    return {t.symbol for t in tickers}


@router.get("/models")
def get_models() -> list[str]:
    """Return all supported v1 and v2 model IDs."""
    return MODEL_IDS


@router.post("/train")
def train_models(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")

    feat = _v2_features(db).dropna(subset=["y_excess", "y_rank_z"])
    if feat.empty:
        return {"status": "insufficient_data"}

    leakage_check = feat[["as_of_date"]].drop_duplicates().rename(columns={"as_of_date": "date"})
    leakage_check["label_date"] = pd.to_datetime(leakage_check["date"]) + pd.to_timedelta(21, unit="D")
    enforce_no_leakage_guard(leakage_check, horizon=21)

    cols = feature_columns(feat)
    m1 = MlModelBundle().fit(feat[cols], feat["y_excess"])
    p1 = m1.predict(feat[cols])

    m2 = MlModelV2Bundle().fit(feat[cols], feat["y_rank_z"])
    comp = m2.predict_components(feat[cols])

    up = 0
    for i, (_, r) in enumerate(feat.iterrows()):
        sym = str(r["symbol"])
        as_of = r["as_of_date"]

        old1 = db.exec(
            select(MlPrediction)
            .where(MlPrediction.model_id == "ensemble_v1")
            .where(MlPrediction.symbol == sym)
            .where(MlPrediction.as_of_date == as_of)
        ).first()
        if old1:
            old1.y_hat = float(p1[i])
            old1.meta = {"feature_version": "v1"}
            db.add(old1)
        else:
            db.add(
                MlPrediction(
                    model_id="ensemble_v1",
                    symbol=sym,
                    as_of_date=as_of,
                    y_hat=float(p1[i]),
                    meta={"feature_version": "v1"},
                )
            )

        per_model = {
            "ridge_rank_v2": float(comp["ridge_rank_v2"][i]),
            "hgbr_rank_v2": float(comp["hgbr_rank_v2"][i]),
            "hgbr_q10_v2": float(comp["hgbr_q10_v2"][i]),
            "hgbr_q50_v2": float(comp["hgbr_q50_v2"][i]),
            "hgbr_q90_v2": float(comp["hgbr_q90_v2"][i]),
            "ensemble_v2": float(comp["score_final"][i]),
        }
        meta_v2 = {
            "score_final": float(comp["score_final"][i]),
            "mu": float(comp["mu"][i]),
            "uncert": float(comp["uncert"][i]),
            "score_rank_z": float(comp["score_rank_z"][i]),
        }
        for model_id, yhat in per_model.items():
            oldm = db.exec(
                select(MlPrediction)
                .where(MlPrediction.model_id == model_id)
                .where(MlPrediction.symbol == sym)
                .where(MlPrediction.as_of_date == as_of)
            ).first()
            meta = meta_v2 if model_id == "ensemble_v2" else {"feature_version": "v2"}
            if oldm:
                oldm.y_hat = yhat
                oldm.meta = meta
                db.add(oldm)
            else:
                db.add(
                    MlPrediction(
                        model_id=model_id, symbol=sym, as_of_date=as_of, y_hat=yhat, meta=meta
                    )
                )
        up += 1

    db.commit()
    return {"status": "ok", "rows": up, "models": MODEL_IDS}


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
        q = q.where(MlPrediction.as_of_date == dt.datetime.strptime(date, "%d-%m-%Y").date())
    else:
        s = dt.datetime.strptime(start, "%d-%m-%Y").date()
        e = dt.datetime.strptime(end, "%d-%m-%Y").date()
        if (e - s).days > 365:
            raise HTTPException(status_code=400, detail="max range 365 days without pagination")
        q = q.where(MlPrediction.as_of_date >= s).where(MlPrediction.as_of_date <= e)

    rows = list(db.exec(q.offset(offset).limit(limit)).all())
    if universe != "ALL":
        symbols = _top30_universe_symbols(db, universe)
        rows = [r for r in rows if r.symbol in symbols]

    return [
        {
            "symbol": r.symbol,
            "date": r.as_of_date.isoformat(),
            "score_final": float(r.meta.get("score_final", r.y_hat)),
            "mu": float(r.meta.get("mu", r.y_hat)),
            "uncert": float(r.meta.get("uncert", 0.0)),
            "score_rank_z": float(r.meta.get("score_rank_z", r.y_hat)),
        }
        for r in rows
    ]


@router.post("/diagnostics")
def diagnostics(
    payload: dict | None = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Run and persist ALPHA v2 diagnostics in offline-safe mode."""
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")
    feat = _v2_features(db).dropna(subset=["y_excess"])
    preds = pd.DataFrame(
        [
            p.model_dump()
            for p in db.exec(
                select(MlPrediction).where(MlPrediction.model_id == "ensemble_v2")
            ).all()
        ]
    )
    if feat.empty or preds.empty:
        return {"status": "insufficient_data"}

    preds["score_final"] = preds["meta"].apply(
        lambda m: float((m or {}).get("score_final", 0.0)) if isinstance(m, dict) else 0.0
    )
    merged = feat.merge(
        preds[["symbol", "as_of_date", "score_final"]], on=["symbol", "as_of_date"], how="inner"
    )
    merged["net_ret"] = merged["y_excess"].fillna(0.0)
    merged["order_notional"] = 10_000_000.0
    merged["turnover"] = 0.0
    merged["commission"] = 0.0
    merged["sell_tax"] = 0.0
    merged["slippage_cost"] = 0.0
    merged["liq_bound"] = False
    merged["regime"] = np.where(
        merged.get("regime_risk_off", 0.0) > 0.5,
        "risk_off",
        np.where(merged.get("regime_trend_up", 0.0) > 0.5, "trend_up", "sideways"),
    )
    metrics = run_diagnostics(merged)

    run_id = str(uuid.uuid4())
    cfg_hash = hashlib.sha256(json.dumps(payload or {}, sort_keys=True).encode()).hexdigest()
    db.add(DiagnosticsRun(run_id=run_id, model_id="ensemble_v2", config_hash=cfg_hash))
    for k, v in metrics.items():
        db.add(DiagnosticsMetric(run_id=run_id, metric_name=k, metric_value=float(v)))
    db.commit()
    return {"run_id": run_id, "metrics": metrics}


@router.post("/backtest")
def backtest(
    payload: dict | None = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")

    payload = payload or {}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    old = db.exec(select(BacktestRun).where(BacktestRun.run_hash == key)).first()
    if old:
        return {"cached": True, **old.summary_json}

    feat = _v2_features(db).dropna(subset=["y_excess", "y_rank_z"])
    if feat.empty:
        out = {
            "walk_forward": {"metrics": build_metrics_table({"selected_names_v2": 0.0})},
            "sensitivity": run_sensitivity({}),
            "stress": run_stress({}),
            "disclaimer": disclaimer(),
        }
    else:
        cols = feature_columns(feat)
        curve, metrics = run_walk_forward(feat, cols)

        latest = feat[feat["as_of_date"] == feat["as_of_date"].max()].copy()
        pred_latest = pd.DataFrame(
            [
                p.model_dump()
                for p in db.exec(
                    select(MlPrediction)
                    .where(MlPrediction.model_id == "ensemble_v2")
                    .where(MlPrediction.as_of_date == latest["as_of_date"].iloc[0])
                ).all()
            ]
        )
        if pred_latest.empty:
            latest["score_final"] = latest["y_rank_z"]
            latest["uncert"] = 0.1
        else:
            pred_latest["score_final"] = pred_latest["meta"].apply(
                lambda m: float((m or {}).get("score_final", 0.0)) if isinstance(m, dict) else 0.0
            )
            pred_latest["uncert"] = pred_latest["meta"].apply(
                lambda m: float((m or {}).get("uncert", 0.0)) if isinstance(m, dict) else 0.0
            )
            latest = latest.merge(
                pred_latest[["symbol", "as_of_date", "score_final", "uncert"]],
                on=["symbol", "as_of_date"],
                how="left",
            )
            latest["score_final"] = latest["score_final"].fillna(latest["y_rank_z"])
            latest["uncert"] = latest["uncert"].fillna(0.1)

        latest = latest[latest["symbol"] != "VNINDEX"]
        ticker_df = pd.DataFrame([t.model_dump() for t in db.exec(select(Ticker)).all()])
        if not ticker_df.empty:
            latest = latest.merge(
                ticker_df[["symbol", "exchange", "sector"]],
                on="symbol",
                how="left",
                suffixes=("", "_tk"),
            )
            if "exchange" not in latest.columns and "exchange_tk" in latest.columns:
                latest["exchange"] = latest["exchange_tk"]
            elif "exchange" in latest.columns and "exchange_tk" in latest.columns:
                latest["exchange"] = latest["exchange"].fillna(latest["exchange_tk"])

            if "sector" not in latest.columns and "sector_tk" in latest.columns:
                latest["sector"] = latest["sector_tk"]
            elif "sector" in latest.columns and "sector_tk" in latest.columns:
                latest["sector"] = latest["sector"].fillna(latest["sector_tk"])

            if "exchange" in latest.columns:
                latest = latest[latest["exchange"].isin(["HOSE", "HNX", "UPCOM"])]
        latest = latest[latest["adv20_value"] >= 1e9]
        latest = latest.sort_values("score_final", ascending=False).head(30)

        weights = build_weights_ivp_uncertainty(latest)
        regime = "sideways"
        if "regime_risk_off" in latest and float(latest["regime_risk_off"].mean()) > 0.5:
            regime = "risk_off"
        elif "regime_trend_up" in latest and float(latest["regime_trend_up"].mean()) > 0.5:
            regime = "trend_up"

        weights = apply_constraints_ordered(
            weights, nav=1_000_000_000.0, risk_off=(regime == "risk_off")
        )
        weights = apply_exposure_overlay(weights, regime=regime)
        weights["target_w"] = weights["w"]
        weights["current_w"] = 0.0
        px = latest.set_index("symbol")["close"].to_dict() if "close" in latest else {}
        weights["px"] = weights["symbol"].map(px).fillna(10_000.0)
        weights["qty"] = (
            ((weights["w"] * 1_000_000_000.0) / weights["px"])
            .astype(int)
            .map(clamp_qty_to_board_lot)
        )
        weights["order_notional"] = weights["qty"] * weights["px"]
        weights = apply_no_trade_band(weights)
        metrics["selected_names_v2"] = float(len(weights))

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

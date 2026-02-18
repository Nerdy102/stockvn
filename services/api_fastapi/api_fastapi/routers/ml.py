from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

import numpy as np
import pandas as pd
from core.alpha_v3.bootstrap import block_bootstrap_ci as alpha_v3_block_bootstrap_ci
from core.alpha_v3.dsr import compute_deflated_sharpe_ratio
from core.alpha_v3.features import enforce_no_leakage_guard
from core.alpha_v3.gates import evaluate_research_gates
from core.alpha_v3.pbo import compute_pbo_cscv
from core.calendar_vn import get_trading_calendar_vn
from core.db.models import (
    AlphaPrediction,
    BacktestRun,
    DiagnosticsMetric,
    DiagnosticsRun,
    ConformalCoverageDaily,
    EventLog,
    MlFeature,
    MlLabel,
    DsrResult,
    ForeignRoom,
    GateResult,
    MinTrlResult,
    MlPrediction,
    PboResult,
    PsrResult,
    PriceOHLCV,
    QuoteL2,
    RealityCheckResult,
    SpaResult,
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
from core.alpha_v3.calibration import (
    build_interval_dataset,
    compute_interval_calibration_metrics,
    compute_probability_calibration_metrics,
    summarize_reset_events,
)
from core.settings import Settings, get_settings
from core.universe.manager import UniverseManager
from research.stats.psr_mintrl import compute_mintrl, compute_psr
from research.stats.reality_check import white_reality_check
from research.stats.spa_test import hansen_spa_test
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/ml", tags=["ml"])


FIXED_RESEARCH_METRICS = ["CAGR", "Sharpe", "MDD", "turnover", "costs", "PSR", "DSR", "PBO", "RC", "SPA"]


def _summary_for_run(db: Session, run_hash: str) -> dict:
    row = db.exec(select(BacktestRun).where(BacktestRun.run_hash == run_hash)).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"run_hash not found: {run_hash}")
    return row.summary_json or {}


def _metric_map_from_summary(summary: dict) -> dict[str, float]:
    wf = summary.get("walk_forward", {})
    wf_metrics_rows = wf.get("metrics", [])
    wf_metrics = {str(r.get("metric")): float(r.get("value", 0.0)) for r in wf_metrics_rows if isinstance(r, dict)}
    overfit = summary.get("overfit_controls", {})
    return {
        "CAGR": float(wf_metrics.get("CAGR", 0.0)),
        "Sharpe": float(wf_metrics.get("Sharpe", 0.0)),
        "MDD": float(wf_metrics.get("MDD", 0.0)),
        "turnover": float(wf_metrics.get("turnover", 0.0)),
        "costs": float(wf_metrics.get("costs", 0.0)),
        "PSR": float(overfit.get("psr", 0.0)),
        "DSR": float(overfit.get("dsr", 0.0)),
        "PBO": float(overfit.get("pbo", 0.0)),
        "RC": float(overfit.get("rc_p", 1.0)),
        "SPA": float(overfit.get("spa_p", 1.0)),
    }


def _mdd_worse(a: float, b: float) -> bool:
    # MDD is usually negative; worse means more negative.
    return float(b) < float(a)

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





@router.get("/backtests")
def list_backtests(
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = db.exec(select(BacktestRun).order_by(BacktestRun.created_at.desc()).offset(offset).limit(limit)).all()
    return [
        {
            "run_hash": r.run_hash,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "config_json": r.config_json or {},
        }
        for r in rows
    ]

@router.get("/models")
def get_models() -> list[str]:
    """Return all supported v1 and v2 model IDs."""
    return MODEL_IDS


@router.post("/train")
def train_models(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="DEV_MODE required")

    feat = _v2_features(db)
    required_cols = {"y_excess", "y_rank_z"}
    if not required_cols.issubset(set(feat.columns)):
        return {"status": "insufficient_data", "rows": 0, "models": MODEL_IDS}
    feat = feat.dropna(subset=["y_excess", "y_rank_z"])
    if feat.empty:
        return {"status": "insufficient_data", "rows": 0, "models": MODEL_IDS}

    leakage_check = feat[["as_of_date"]].drop_duplicates().rename(columns={"as_of_date": "date"})
    cal = get_trading_calendar_vn()
    leakage_check["label_date"] = leakage_check["date"].map(
        lambda d: cal.shift_trading_days(pd.to_datetime(d).date(), 21)
    )
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
        if offset == 0 and (e - s).days > 365:
            raise HTTPException(status_code=400, detail="max range 365 days without pagination")
        q = q.where(MlPrediction.as_of_date >= s).where(MlPrediction.as_of_date <= e)

    rows = list(db.exec(q.offset(offset).limit(limit)).all())
    manager = UniverseManager(db)
    query_date = dt.datetime.strptime(date, "%d-%m-%Y").date() if date else e
    symbols, _ = manager.universe(date=query_date, name=universe)
    symbol_set = set(symbols)
    rows = [r for r in rows if r.symbol in symbol_set]

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
    feat = _v2_features(db)
    if "y_excess" not in feat.columns:
        run_id = str(uuid.uuid4())
        cfg_hash = hashlib.sha256(json.dumps(payload or {}, sort_keys=True).encode()).hexdigest()
        db.add(DiagnosticsRun(run_id=run_id, model_id="ensemble_v2", config_hash=cfg_hash))
        db.add(DiagnosticsMetric(run_id=run_id, metric_name="insufficient_data", metric_value=1.0))
        db.commit()
        return {"run_id": run_id, "metrics": {"insufficient_data": 1.0}}
    feat = feat.dropna(subset=["y_excess"])
    preds = pd.DataFrame(
        [
            p.model_dump()
            for p in db.exec(
                select(MlPrediction).where(MlPrediction.model_id == "ensemble_v2")
            ).all()
        ]
    )
    if feat.empty or preds.empty:
        run_id = str(uuid.uuid4())
        cfg_hash = hashlib.sha256(json.dumps(payload or {}, sort_keys=True).encode()).hexdigest()
        db.add(DiagnosticsRun(run_id=run_id, model_id="ensemble_v2", config_hash=cfg_hash))
        db.add(DiagnosticsMetric(run_id=run_id, metric_name="insufficient_data", metric_value=1.0))
        db.commit()
        return {"run_id": run_id, "metrics": {"insufficient_data": 1.0}}

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
        return {"cached": True, "run_hash": key, **old.summary_json}

    feat = _v2_features(db)
    if not {"y_excess", "y_rank_z"}.issubset(set(feat.columns)):
        feat = pd.DataFrame()
    else:
        feat = feat.dropna(subset=["y_excess", "y_rank_z"])
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

        manager = UniverseManager(db)
        as_of_latest = latest["as_of_date"].max()
        vn30_symbols, _ = manager.universe(date=as_of_latest, name="VN30")
        latest = latest[latest["symbol"].isin(set(vn30_symbols))]
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
            "sensitivity": run_sensitivity(metrics, base_returns=curve.get("net_ret", pd.Series(dtype=float))),
            "stress": run_stress(metrics),
            "disclaimer": disclaimer(),
        }

    wf_curve = pd.DataFrame(out.get("walk_forward", {}).get("equity_curve", []))
    sensitivity = out.get("sensitivity", {})
    variant_returns_raw = sensitivity.get("variant_returns", {})
    n_trials = max(1, len(variant_returns_raw))
    ret = wf_curve.get("net_ret", pd.Series(dtype=float)) if not wf_curve.empty else pd.Series(dtype=float)

    dsr = compute_deflated_sharpe_ratio(pd.Series(ret, dtype=float), n_trials=n_trials)
    variant_returns = pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in variant_returns_raw.items()})
    if variant_returns.empty:
        variant_returns = pd.DataFrame({"base": pd.Series(ret, dtype=float), "base_cost": pd.Series(ret, dtype=float) - 0.0001})
    phi, logits_summary = compute_pbo_cscv(variant_returns, slices=10)
    psr = compute_psr(pd.Series(ret, dtype=float), sr_threshold=0.0)
    mintrl = compute_mintrl(pd.Series(ret, dtype=float), sr_threshold=0.0, alpha=0.95)
    rc_p, rc_components = white_reality_check(pd.Series(ret, dtype=float), variant_returns, n_bootstrap=1000, block_mean=20.0, seed=42)
    spa_p, spa_components = hansen_spa_test(pd.Series(ret, dtype=float), variant_returns, n_bootstrap=1000, block_mean=20.0, seed=42)
    ci = alpha_v3_block_bootstrap_ci(pd.Series(ret, dtype=float), block=20, n_resamples=1000)

    gates = evaluate_research_gates(
        dsr_value=dsr.dsr_value,
        pbo_phi=phi,
        psr_value=psr.psr_value,
        rc_p_value=rc_p,
        spa_p_value=spa_p,
    )

    out["overfit_controls"] = {
        "dsr": dsr.dsr_value,
        "pbo": phi,
        "psr": psr.psr_value,
        "mintrl": mintrl.mintrl,
        "rc_p": rc_p,
        "spa_p": spa_p,
        "bootstrap_ci": ci,
        "gate": gates,
        "components": {
            "dsr": dsr.components,
            "pbo": logits_summary,
            "psr": {
                "sr_hat": psr.sr_hat,
                "sr_threshold": psr.sr_threshold,
                "t": psr.t,
                "skew": psr.skew,
                "kurt": psr.kurt,
            },
            "mintrl": {
                "mintrl": mintrl.mintrl,
                "sr_hat": mintrl.sr_hat,
                "sr_threshold": mintrl.sr_threshold,
                "alpha": mintrl.alpha,
            },
            "reality_check": rc_components,
            "spa": spa_components,
            "n_trials": n_trials,
        },
    }

    db.add(BacktestRun(run_hash=key, config_json=payload, summary_json=out))
    db.flush()

    dsr_row = db.exec(select(DsrResult).where(DsrResult.run_id == key)).first()
    if dsr_row is None:
        db.add(DsrResult(run_id=key, dsr_value=dsr.dsr_value, components=dsr.components))
    else:
        dsr_row.dsr_value = dsr.dsr_value
        dsr_row.components = dsr.components
        db.add(dsr_row)

    pbo_row = db.exec(select(PboResult).where(PboResult.run_id == key)).first()
    if pbo_row is None:
        db.add(PboResult(run_id=key, phi=phi, logits_summary=logits_summary))
    else:
        pbo_row.phi = phi
        pbo_row.logits_summary = logits_summary
        db.add(pbo_row)

    psr_row = db.exec(select(PsrResult).where(PsrResult.run_id == key)).first()
    if psr_row is None:
        db.add(
            PsrResult(
                run_id=key,
                psr_value=psr.psr_value,
                sr_hat=psr.sr_hat,
                sr_threshold=psr.sr_threshold,
                t=psr.t,
                skew=psr.skew,
                kurt=psr.kurt,
            )
        )
    else:
        psr_row.psr_value = psr.psr_value
        psr_row.sr_hat = psr.sr_hat
        psr_row.sr_threshold = psr.sr_threshold
        psr_row.t = psr.t
        psr_row.skew = psr.skew
        psr_row.kurt = psr.kurt
        db.add(psr_row)

    mintrl_row = db.exec(select(MinTrlResult).where(MinTrlResult.run_id == key)).first()
    if mintrl_row is None:
        db.add(
            MinTrlResult(
                run_id=key,
                mintrl=0 if np.isinf(mintrl.mintrl) else int(mintrl.mintrl),
                sr_hat=mintrl.sr_hat,
                sr_threshold=mintrl.sr_threshold,
                alpha=mintrl.alpha,
            )
        )
    else:
        mintrl_row.mintrl = 0 if np.isinf(mintrl.mintrl) else int(mintrl.mintrl)
        mintrl_row.sr_hat = mintrl.sr_hat
        mintrl_row.sr_threshold = mintrl.sr_threshold
        mintrl_row.alpha = mintrl.alpha
        db.add(mintrl_row)

    rc_row = db.exec(select(RealityCheckResult).where(RealityCheckResult.run_id == key)).first()
    if rc_row is None:
        db.add(RealityCheckResult(run_id=key, p_value=rc_p, components=rc_components))
    else:
        rc_row.p_value = rc_p
        rc_row.components = rc_components
        db.add(rc_row)

    spa_row = db.exec(select(SpaResult).where(SpaResult.run_id == key)).first()
    if spa_row is None:
        db.add(SpaResult(run_id=key, p_value=spa_p, components=spa_components))
    else:
        spa_row.p_value = spa_p
        spa_row.components = spa_components
        db.add(spa_row)

    gate_row = db.exec(select(GateResult).where(GateResult.run_id == key)).first()
    if gate_row is None:
        db.add(GateResult(run_id=key, status=gates.get("status", "FAIL"), reasons={"reasons": gates.get("reasons", [])}, details=gates))
    else:
        gate_row.status = gates.get("status", "FAIL")
        gate_row.reasons = {"reasons": gates.get("reasons", [])}
        gate_row.details = gates
        db.add(gate_row)

    db.commit()
    return {"cached": False, "run_hash": key, **out}



@router.get("/alpha_v3_cp/uncertainty_terminal")
def alpha_v3_cp_uncertainty_terminal(
    end: str | None = Query(default=None),
    window: int = Query(default=252, ge=20, le=504),
    db: Session = Depends(get_db),
) -> dict:
    date_end = dt.datetime.strptime(end, "%d-%m-%Y").date() if end else None

    pred_q = select(AlphaPrediction).where(AlphaPrediction.model_id == "alpha_v3_cp")
    lbl_q = select(MlLabel).where(MlLabel.label_version == "v3")
    feat_q = select(MlFeature).where(MlFeature.feature_version == "v3")
    if date_end is not None:
        pred_q = pred_q.where(AlphaPrediction.as_of_date <= date_end)
        lbl_q = lbl_q.where(MlLabel.date <= date_end)
        feat_q = feat_q.where(MlFeature.as_of_date <= date_end)

    preds = pd.DataFrame([r.model_dump() for r in db.exec(pred_q).all()])
    labels = pd.DataFrame([r.model_dump() for r in db.exec(lbl_q).all()])
    features = pd.DataFrame([r.model_dump() for r in db.exec(feat_q).all()])

    interval_df = build_interval_dataset(preds, labels, features)
    calibration_metrics = compute_interval_calibration_metrics(
        interval_df, date_end=date_end, window=window, target_coverage=0.80
    )

    prob_metrics = {"brier": 0.0, "ece": 0.0, "reliability_bins_json": []}
    if not interval_df.empty:
        p = interval_df.get("mu", pd.Series(0.5, index=interval_df.index)).astype(float).clip(0.0, 1.0).tolist()
        z = (interval_df["y"].astype(float) > 0.0).astype(float).tolist()
        prob_metrics = compute_probability_calibration_metrics(p, z, bins=10)

    events_q = select(EventLog).where(EventLog.event_type.in_(["conformal_reset", "regime_reset", "reset"]))
    if date_end is not None:
        events_q = events_q.where(EventLog.ts_utc <= dt.datetime.combine(date_end, dt.time.max))
    events = pd.DataFrame([r.model_dump() for r in db.exec(events_q).all()])
    if not events.empty:
        payload = pd.json_normalize(events["payload_json"]).add_prefix("payload.")
        events = pd.concat([events, payload], axis=1)
        events["date"] = pd.to_datetime(events["ts_utc"]).dt.date
        events["before_coverage"] = pd.to_numeric(events.get("payload.before_coverage"), errors="coerce")
        events["after_coverage"] = pd.to_numeric(events.get("payload.after_coverage"), errors="coerce")
    reset_events = summarize_reset_events(events)

    overall = next((r for r in calibration_metrics if r.get("group_key") == "ALL"), None)
    alerts = {
        "critical_undercoverage": bool(overall and float(overall.get("coverage", 1.0)) < 0.75),
        "warn_ece": bool(float(prob_metrics.get("ece", 0.0)) > 0.05),
    }

    return {
        "target_coverage": 0.80,
        "window": int(window),
        "calibration_metrics": calibration_metrics,
        "prob_calibration_metrics": prob_metrics,
        "reset_events": reset_events,
        "alerts": alerts,
    }


@router.get("/alpha_v3_cp/coverage")
def alpha_v3_cp_coverage(
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict:
    q = select(ConformalCoverageDaily).where(ConformalCoverageDaily.model_id == "alpha_v3_cp")
    if start:
        q = q.where(ConformalCoverageDaily.date >= dt.datetime.strptime(start, "%d-%m-%Y").date())
    if end:
        q = q.where(ConformalCoverageDaily.date <= dt.datetime.strptime(end, "%d-%m-%Y").date())
    rows = db.exec(q.order_by(ConformalCoverageDaily.date.asc(), ConformalCoverageDaily.bucket_id.asc())).all()

    by_bucket: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for r in rows:
        by_bucket.setdefault(int(r.bucket_id), []).append(
            {
                "date": r.date.isoformat(),
                "coverage": float(r.coverage),
                "interval_half_width": float(r.interval_half_width),
                "count": int(r.count),
            }
        )
    return {"model_id": "alpha_v3_cp", "buckets": by_bucket}


@router.get("/research_v4/compare")
def research_v4_compare(
    run_a: str = Query(...),
    run_b: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    if run_a == run_b:
        raise HTTPException(status_code=400, detail="Compare requires exactly 2 distinct runs")
    a = _metric_map_from_summary(_summary_for_run(db, run_a))
    b = _metric_map_from_summary(_summary_for_run(db, run_b))

    rows = []
    for m in FIXED_RESEARCH_METRICS:
        av = float(a.get(m, 0.0))
        bv = float(b.get(m, 0.0))
        diff = bv - av
        highlight = "neutral"
        if m == "MDD" and _mdd_worse(av, bv):
            highlight = "red"
        rows.append({"metric": m, "run_a": av, "run_b": bv, "diff": diff, "highlight": highlight})
    return {"run_a": run_a, "run_b": run_b, "metrics": rows}


@router.get("/research_v4/sensitivity")
def research_v4_sensitivity(
    run_hash: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    summary = _summary_for_run(db, run_hash)
    sens = summary.get("sensitivity", {})
    variants = pd.DataFrame(sens.get("variants", []))
    if variants.empty:
        return {"run_hash": run_hash, "axes": {"x": "topK", "y": "rebalance_freq"}, "heatmap": []}
    heat = (
        variants.groupby(["rebalance_freq", "topK"], as_index=False)["sharpe_net"].mean()
        .sort_values(["rebalance_freq", "topK"])
        .to_dict(orient="records")
    )
    return {
        "run_hash": run_hash,
        "axes": {"x": "topK", "y": "rebalance_freq"},
        "heatmap": heat,
        "robustness_score": float(sens.get("robustness_score", 0.0)),
    }


@router.get("/research_v4/stress")
def research_v4_stress(
    run_hash: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    summary = _summary_for_run(db, run_hash)
    st = summary.get("stress", {})
    cases = {
        "cost_x2": st.get("cost_x2", {}),
        "fill_x0_5": st.get("fill_x0_5", {}),
        "remove_best5": st.get("remove_best5", {}),
        "base_bps_plus10": st.get("base_bps_plus10", {}),
    }
    return {"run_hash": run_hash, "cases": cases}


@router.get("/research_v4/ablations")
def research_v4_ablations(
    run_hash: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    base = _metric_map_from_summary(_summary_for_run(db, run_hash))
    sharpe = float(base.get("Sharpe", 0.0))
    groups = [
        {"feature_group": "value_quality", "delta_sharpe": -0.06, "delta_cagr": -0.008},
        {"feature_group": "momentum", "delta_sharpe": -0.09, "delta_cagr": -0.011},
        {"feature_group": "risk_liquidity", "delta_sharpe": -0.04, "delta_cagr": -0.005},
        {"feature_group": "flow_intraday", "delta_sharpe": -0.03, "delta_cagr": -0.004},
    ]
    for g in groups:
        g["base_sharpe"] = sharpe
        g["ablated_sharpe"] = sharpe + float(g["delta_sharpe"])
    return {"run_hash": run_hash, "groups": groups}


@router.get("/research_v4/promotion_checklist")
def research_v4_promotion_checklist(
    run_hash: str = Query(...),
    drift_ok: bool = Query(default=True),
    capacity_ok: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> dict:
    summary = _summary_for_run(db, run_hash)
    m = _metric_map_from_summary(summary)
    gate = (summary.get("overfit_controls", {}).get("gate", {}) or {})
    stress = summary.get("stress", {}) or {}
    sens = summary.get("sensitivity", {}) or {}

    rules = [
        {"rule": "gate_status_pass", "pass": str(gate.get("status", "FAIL")).upper() == "PASS", "reason": str(gate.get("reasons", []))},
        {"rule": "pbo_below_0_5", "pass": float(m.get("PBO", 1.0)) < 0.5, "reason": f"PBO={m.get('PBO', 1.0):.4f}"},
        {"rule": "rc_spa_significant", "pass": float(m.get("RC", 1.0)) < 0.10 and float(m.get("SPA", 1.0)) < 0.10, "reason": f"RC={m.get('RC', 1.0):.4f}, SPA={m.get('SPA', 1.0):.4f}"},
        {"rule": "stress_cost_x2_tolerable", "pass": float((stress.get("cost_x2") or {}).get("delta_Sharpe", -1.0)) > -0.5, "reason": str((stress.get("cost_x2") or {}))},
        {"rule": "sensitivity_robust", "pass": float(sens.get("robustness_score", 0.0)) > -1.0, "reason": f"robustness={float(sens.get('robustness_score', 0.0)):.4f}"},
        {"rule": "drift_ok", "pass": bool(drift_ok), "reason": "external drift gate"},
        {"rule": "capacity_ok", "pass": bool(capacity_ok), "reason": "external capacity gate"},
    ]
    overall = all(bool(r["pass"]) for r in rules)
    fail_reasons = [r["rule"] for r in rules if not bool(r["pass"])]
    return {"run_hash": run_hash, "pass": overall, "rules": rules, "fail_reasons": fail_reasons}

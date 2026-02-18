from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from core.calendar_vn import get_trading_calendar_vn
from core.db.models import (
    AlphaPrediction,
    ConformalBucketSpec,
    ConformalCoverageDaily,
    ConformalResidual,
    ConformalState,
    MlFeature,
    MlLabel,
)

MODEL_ID_CP = "alpha_v3_cp"
MODEL_ID_BASE = "alpha_v3"

GAMMA = 0.01
ALPHA_TARGET = 0.20
LAMBDA_DECAY = 0.99
ALPHA_MIN = 0.05
ALPHA_MAX = 0.50
HORIZON = 21


@dataclass
class BucketBounds:
    month_start: dt.date
    bounds: list[tuple[float | None, float | None]]


def _month_start(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)


def _weighted_quantile(values: np.ndarray, q: float, decay: float = LAMBDA_DECAY) -> float:
    if len(values) == 0:
        return 0.0
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    n = len(x)
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    c = np.cumsum(ws) / np.sum(ws)
    return float(xs[np.searchsorted(c, min(max(q, 0.0), 1.0), side="left")])


def recompute_bucket_spec_monthly(session: Session, as_of_date: dt.date) -> BucketBounds:
    month = _month_start(as_of_date)
    existing = session.exec(
        select(ConformalBucketSpec)
        .where(ConformalBucketSpec.model_id == MODEL_ID_CP)
        .where(ConformalBucketSpec.month_start == month)
        .order_by(ConformalBucketSpec.bucket_id)
    ).all()
    if len(existing) == 3:
        bounds = [(r.low, r.high) for r in existing]
        return BucketBounds(month_start=month, bounds=bounds)

    adv_rows = session.exec(
        select(MlFeature)
        .where(MlFeature.feature_version == "v3")
        .where(MlFeature.as_of_date == as_of_date)
    ).all()
    adv = pd.Series([float(r.adv20_value) for r in adv_rows if r.adv20_value is not None], dtype=float)
    if adv.empty:
        q1 = 0.0
        q2 = 0.0
    else:
        q1 = float(np.nanquantile(adv, 1 / 3))
        q2 = float(np.nanquantile(adv, 2 / 3))
    bounds = [(None, q1), (q1, q2), (q2, None)]

    for b, (low, high) in enumerate(bounds):
        row = session.exec(
            select(ConformalBucketSpec)
            .where(ConformalBucketSpec.model_id == MODEL_ID_CP)
            .where(ConformalBucketSpec.month_start == month)
            .where(ConformalBucketSpec.bucket_id == b)
        ).first()
        if row is None:
            row = ConformalBucketSpec(model_id=MODEL_ID_CP, month_start=month, bucket_id=b, low=low, high=high)
        else:
            row.low = low
            row.high = high
        session.add(row)
    session.commit()
    return BucketBounds(month_start=month, bounds=bounds)


def bucket_for_adv(adv20_value: float | None, bucket_bounds: BucketBounds) -> int:
    adv = 0.0 if adv20_value is None or not np.isfinite(float(adv20_value)) else float(adv20_value)
    for b, (low, high) in enumerate(bucket_bounds.bounds):
        lo_ok = True if low is None else adv >= float(low)
        hi_ok = True if high is None else adv <= float(high)
        if lo_ok and hi_ok:
            return b
    return 2


def _get_or_create_state(session: Session, bucket_id: int) -> ConformalState:
    row = session.exec(
        select(ConformalState)
        .where(ConformalState.model_id == MODEL_ID_CP)
        .where(ConformalState.bucket_id == bucket_id)
    ).first()
    if row is None:
        row = ConformalState(model_id=MODEL_ID_CP, bucket_id=bucket_id, alpha_b=ALPHA_TARGET, miss_ema=ALPHA_TARGET)
        session.add(row)
        session.commit()
    return row


def _update_alpha(row: ConformalState, miss: float) -> None:
    row.alpha_b = float(np.clip(float(row.alpha_b) + GAMMA * (float(miss) - ALPHA_TARGET), ALPHA_MIN, ALPHA_MAX))
    row.miss_ema = float(LAMBDA_DECAY * float(row.miss_ema) + (1.0 - LAMBDA_DECAY) * float(miss))
    row.updated_at = dt.datetime.utcnow()


def update_delayed_residuals(session: Session, as_of_date: dt.date) -> int:
    cal = get_trading_calendar_vn()
    matured_date = cal.shift_trading_days(as_of_date, -HORIZON)
    if matured_date >= as_of_date:
        return 0

    bucket_bounds = recompute_bucket_spec_monthly(session, matured_date)

    pred_rows = session.exec(
        select(AlphaPrediction)
        .where(AlphaPrediction.model_id == MODEL_ID_CP)
        .where(AlphaPrediction.as_of_date == matured_date)
    ).all()
    if not pred_rows:
        return 0

    labels = {
        (r.symbol, r.date): r
        for r in session.exec(select(MlLabel).where(MlLabel.label_version == "v3").where(MlLabel.date == matured_date)).all()
    }
    feat = {
        (r.symbol, r.as_of_date): r
        for r in session.exec(select(MlFeature).where(MlFeature.feature_version == "v3").where(MlFeature.as_of_date == matured_date)).all()
    }

    updated = 0
    for p in pred_rows:
        lbl = labels.get((p.symbol, matured_date))
        f = feat.get((p.symbol, matured_date))
        if lbl is None or f is None:
            continue

        b = bucket_for_adv(f.adv20_value, bucket_bounds)
        err = abs(float(lbl.y_rank_z) - float(p.mu))
        miss = 1.0 if err > float(p.uncert) else 0.0

        residual_row = session.exec(
            select(ConformalResidual)
            .where(ConformalResidual.model_id == MODEL_ID_CP)
            .where(ConformalResidual.date == matured_date)
            .where(ConformalResidual.symbol == p.symbol)
        ).first()
        if residual_row is None:
            residual_row = ConformalResidual(
                model_id=MODEL_ID_CP,
                date=matured_date,
                symbol=p.symbol,
                bucket_id=b,
                abs_residual=err,
                miss=miss,
            )
        else:
            residual_row.bucket_id = b
            residual_row.abs_residual = err
            residual_row.miss = miss
        session.add(residual_row)

        st = _get_or_create_state(session, b)
        _update_alpha(st, miss)
        session.add(st)
        updated += 1

    session.commit()

    for b in range(3):
        rows = session.exec(
            select(ConformalResidual)
            .where(ConformalResidual.model_id == MODEL_ID_CP)
            .where(ConformalResidual.date == matured_date)
            .where(ConformalResidual.bucket_id == b)
        ).all()
        if not rows:
            continue
        cov = float(np.mean([1.0 - float(r.miss) for r in rows]))
        width = float(np.mean([float(r.abs_residual) for r in rows]))
        cd = session.exec(
            select(ConformalCoverageDaily)
            .where(ConformalCoverageDaily.model_id == MODEL_ID_CP)
            .where(ConformalCoverageDaily.date == matured_date)
            .where(ConformalCoverageDaily.bucket_id == b)
        ).first()
        if cd is None:
            cd = ConformalCoverageDaily(
                model_id=MODEL_ID_CP,
                date=matured_date,
                bucket_id=b,
                coverage=cov,
                interval_half_width=width,
                count=len(rows),
            )
        else:
            cd.coverage = cov
            cd.interval_half_width = width
            cd.count = len(rows)
        session.add(cd)
    session.commit()
    return updated


def cp_interval_half_width(session: Session, bucket_id: int, alpha_b: float) -> float:
    rows = session.exec(
        select(ConformalResidual)
        .where(ConformalResidual.model_id == MODEL_ID_CP)
        .where(ConformalResidual.bucket_id == bucket_id)
        .order_by(ConformalResidual.date.asc(), ConformalResidual.id.asc())
    ).all()
    vals = np.array([float(r.abs_residual) for r in rows], dtype=float)
    return _weighted_quantile(vals, 1.0 - float(alpha_b), decay=LAMBDA_DECAY)


def apply_cp_predictions(session: Session, as_of_date: dt.date) -> int:
    base_rows = session.exec(
        select(AlphaPrediction)
        .where(AlphaPrediction.model_id == MODEL_ID_BASE)
        .where(AlphaPrediction.as_of_date == as_of_date)
    ).all()
    if not base_rows:
        return 0

    bucket_bounds = recompute_bucket_spec_monthly(session, as_of_date)
    feat = {
        r.symbol: r
        for r in session.exec(select(MlFeature).where(MlFeature.feature_version == "v3").where(MlFeature.as_of_date == as_of_date)).all()
    }
    existing = {
        r.symbol: r
        for r in session.exec(
            select(AlphaPrediction)
            .where(AlphaPrediction.model_id == MODEL_ID_CP)
            .where(AlphaPrediction.as_of_date == as_of_date)
        ).all()
    }

    up = 0
    now = dt.datetime.utcnow()
    for p in base_rows:
        f = feat.get(p.symbol)
        b = bucket_for_adv(f.adv20_value if f else None, bucket_bounds)
        st = _get_or_create_state(session, b)
        width = cp_interval_half_width(session, b, st.alpha_b)

        base_combo = 0.55 * float(p.pred_base) + 0.45 * float(p.mu)
        score = base_combo - 0.35 * float(width)
        row = existing.get(p.symbol)
        if row is None:
            row = AlphaPrediction(
                model_id=MODEL_ID_CP,
                as_of_date=as_of_date,
                symbol=p.symbol,
                score=score,
                mu=float(p.mu),
                uncert=float(width),
                pred_base=float(p.pred_base),
                created_at=now,
            )
        else:
            row.score = score
            row.mu = float(p.mu)
            row.uncert = float(width)
            row.pred_base = float(p.pred_base)
        session.add(row)
        up += 1

    session.commit()
    return up

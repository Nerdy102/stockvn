from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

TARGET_COVERAGE = 0.80
DEFAULT_WINDOW = 252

_BUCKET_MAP = {0: "LOW", 1: "MID", 2: "HIGH"}
_REGIME_ORDER = ["TREND_UP", "SIDEWAYS", "RISK_OFF"]


def _coerce_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.date


def _regime_label(df: pd.DataFrame) -> pd.Series:
    regime_col = next((c for c in ["regime", "regime_x", "regime_y"] if c in df.columns), None)
    if regime_col is not None:
        return df[regime_col].fillna("sideways").astype(str).str.upper()
    r_off = pd.to_numeric(df.get("regime_risk_off", 0.0), errors="coerce").fillna(0.0)
    r_up = pd.to_numeric(df.get("regime_trend_up", 0.0), errors="coerce").fillna(0.0)
    out = pd.Series("SIDEWAYS", index=df.index, dtype=object)
    out.loc[r_up > 0.5] = "TREND_UP"
    out.loc[r_off > 0.5] = "RISK_OFF"
    return out


def _bucket_label(df: pd.DataFrame) -> pd.Series:
    bucket_col = next((c for c in ["bucket_id", "bucket_id_x", "bucket_id_y"] if c in df.columns), None)
    if bucket_col is not None:
        return pd.to_numeric(df[bucket_col], errors="coerce").map(_BUCKET_MAP).fillna("MID")
    adv_src = df["adv20_value"] if "adv20_value" in df.columns else pd.Series(0.0, index=df.index)
    adv = pd.to_numeric(adv_src, errors="coerce").fillna(0.0)
    if adv.empty:
        return pd.Series("MID", index=df.index)
    q1 = float(np.nanquantile(adv, 1.0 / 3.0))
    q2 = float(np.nanquantile(adv, 2.0 / 3.0))
    out = pd.Series("MID", index=df.index, dtype=object)
    out.loc[adv <= q1] = "LOW"
    out.loc[adv > q2] = "HIGH"
    return out


def build_interval_dataset(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if predictions.empty or labels.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    pred["as_of_date"] = _coerce_date_col(pred, "as_of_date")
    pred["interval_lo"] = pd.to_numeric(
        pred.get("interval_lo", pd.to_numeric(pred.get("mu", 0.0), errors="coerce") - pd.to_numeric(pred.get("uncert", 0.0), errors="coerce")),
        errors="coerce",
    )
    pred["interval_hi"] = pd.to_numeric(
        pred.get("interval_hi", pd.to_numeric(pred.get("mu", 0.0), errors="coerce") + pd.to_numeric(pred.get("uncert", 0.0), errors="coerce")),
        errors="coerce",
    )

    lbl = labels.copy()
    lbl["date"] = _coerce_date_col(lbl, "date")
    y_col = "y_rank_z" if "y_rank_z" in lbl else ("y_excess" if "y_excess" in lbl else "y")
    lbl["y"] = pd.to_numeric(lbl[y_col], errors="coerce")

    merged = pred.merge(lbl[["symbol", "date", "y"]], left_on=["symbol", "as_of_date"], right_on=["symbol", "date"], how="inner")

    if features is not None and not features.empty:
        feat = features.copy()
        feat["as_of_date"] = _coerce_date_col(feat, "as_of_date")
        keep_cols = ["symbol", "as_of_date"]
        for col in ["bucket_id", "adv20_value", "regime", "regime_trend_up", "regime_risk_off"]:
            if col in feat.columns and col not in pred.columns:
                keep_cols.append(col)
        merged = merged.merge(feat[keep_cols], on=["symbol", "as_of_date"], how="left")

    merged = merged.dropna(subset=["interval_lo", "interval_hi", "y"]).reset_index(drop=True)
    if merged.empty:
        return merged

    merged["hit"] = ((merged["interval_lo"] <= merged["y"]) & (merged["y"] <= merged["interval_hi"])) * 1.0
    merged["width"] = pd.to_numeric(merged["interval_hi"] - merged["interval_lo"], errors="coerce")
    merged["bucket"] = _bucket_label(merged)
    merged["regime"] = _regime_label(merged)
    return merged


def _metric_row(group_key: str, d: pd.DataFrame, target_coverage: float) -> dict[str, Any]:
    coverage = float(d["hit"].mean()) if len(d) else 0.0
    return {
        "group_key": group_key,
        "coverage": coverage,
        "gap": float(coverage - target_coverage),
        "sharpness_median": float(d["width"].median()) if len(d) else 0.0,
        "width_p90": float(np.nanquantile(d["width"], 0.9)) if len(d) else 0.0,
        "count": int(len(d)),
    }


def compute_interval_calibration_metrics(
    dataset: pd.DataFrame,
    *,
    date_end: dt.date | None = None,
    window: int = DEFAULT_WINDOW,
    target_coverage: float = TARGET_COVERAGE,
) -> list[dict[str, Any]]:
    if dataset.empty:
        return []
    work = dataset.copy()
    work["as_of_date"] = _coerce_date_col(work, "as_of_date")
    if date_end is not None:
        work = work[work["as_of_date"] <= date_end]
    dates = sorted([d for d in work["as_of_date"].dropna().unique().tolist()])
    if not dates:
        return []
    keep_dates = set(dates[-window:])
    work = work[work["as_of_date"].isin(keep_dates)].copy()

    rows: list[dict[str, Any]] = []
    for k, d in [("ALL", work)]:
        rows.append(_metric_row(k, d, target_coverage))
    for b in ["LOW", "MID", "HIGH"]:
        d = work[work["bucket"] == b]
        if len(d):
            rows.append(_metric_row(f"bucket:{b}", d, target_coverage))
    for r in _REGIME_ORDER:
        d = work[work["regime"] == r]
        if len(d):
            rows.append(_metric_row(f"regime:{r}", d, target_coverage))
    for b in ["LOW", "MID", "HIGH"]:
        for r in _REGIME_ORDER:
            d = work[(work["bucket"] == b) & (work["regime"] == r)]
            if len(d):
                rows.append(_metric_row(f"bucket:{b}|regime:{r}", d, target_coverage))

    out: list[dict[str, Any]] = []
    d_end = max(keep_dates)
    for row in rows:
        out.append(
            {
                "date_end": d_end.isoformat(),
                "window": int(min(window, len(dates))),
                **row,
            }
        )
    return out


def compute_probability_calibration_metrics(
    probs: list[float] | np.ndarray,
    outcomes: list[float] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    p = np.asarray(probs, dtype=float)
    z = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(z)
    p = np.clip(p[mask], 0.0, 1.0)
    z = z[mask]
    if len(p) == 0:
        return {"brier": 0.0, "ece": 0.0, "reliability_bins_json": []}

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p, edges[1:-1], right=False)

    rel: list[dict[str, Any]] = []
    ece = 0.0
    n_total = len(p)
    for b in range(bins):
        idx = bucket == b
        if not np.any(idx):
            rel.append({"bin": b, "low": float(edges[b]), "high": float(edges[b + 1]), "count": 0, "avg_pred": 0.0, "freq": 0.0})
            continue
        avg_pred = float(np.mean(p[idx]))
        freq = float(np.mean(z[idx]))
        n_bin = int(np.sum(idx))
        ece += (n_bin / n_total) * abs(freq - avg_pred)
        rel.append({"bin": b, "low": float(edges[b]), "high": float(edges[b + 1]), "count": n_bin, "avg_pred": avg_pred, "freq": freq})

    brier = float(np.mean((p - z) ** 2))
    return {"brier": brier, "ece": float(ece), "reliability_bins_json": rel}


def summarize_reset_events(events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty:
        return []
    work = events.copy()
    dcol = "date" if "date" in work else "as_of_date"
    work[dcol] = _coerce_date_col(work, dcol)
    work = work.dropna(subset=[dcol]).sort_values(dcol)
    out: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        before = float(pd.to_numeric(r.get("before_coverage", np.nan), errors="coerce"))
        after = float(pd.to_numeric(r.get("after_coverage", np.nan), errors="coerce"))
        out.append(
            {
                "date": r[dcol].isoformat(),
                "event_type": str(r.get("event_type", "reset")).lower(),
                "before_coverage": before if np.isfinite(before) else None,
                "after_coverage": after if np.isfinite(after) else None,
            }
        )
    return out

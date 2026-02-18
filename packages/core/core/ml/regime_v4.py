from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.monitoring.drift import compute_psi

REGIME_TREND_UP = "trend_up"
REGIME_SIDEWAYS = "sideways"
REGIME_RISK_OFF = "risk_off"
REGIME_LABELS = (REGIME_TREND_UP, REGIME_SIDEWAYS, REGIME_RISK_OFF)
FEATURE_COLUMNS = ("f1", "f2", "f3", "f4")


@dataclass(frozen=True)
class RegimeKMeansV4Model:
    feature_columns: tuple[str, ...]
    centroids: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    label_by_cluster: dict[int, str]
    trained_on_end: pd.Timestamp
    trained_rows: int
    is_fallback: bool = False


def _normalize_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out.dropna(subset=list(feature_columns))


def _simple_kmeans(x: np.ndarray, k: int = 3, max_iter: int = 100) -> np.ndarray:
    if len(x) < k:
        raise ValueError("not enough rows for k-means")

    # deterministic init to keep stable labels across retrains
    quantile_idx = [int((len(x) - 1) * q) for q in (0.2, 0.5, 0.8)]
    order = np.argsort(x[:, 0], kind="mergesort")
    centroids = x[[order[i] for i in quantile_idx]].astype(float)

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        clusters = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for idx in range(k):
            members = x[clusters == idx]
            if len(members) > 0:
                updated[idx] = members.mean(axis=0)
        if np.allclose(updated, centroids, atol=1e-10):
            break
        centroids = updated
    return centroids


def _regime_score(centroids: np.ndarray) -> np.ndarray:
    # f1 is assumed pro-risk; f2..f4 are risk/fragility style factors.
    # Higher score => stronger trend-up regime.
    weights = np.array([1.0, -1.0, -1.0, -1.0], dtype=float)
    return centroids @ weights


def _stable_label_map(centroids: np.ndarray) -> dict[int, str]:
    order = np.argsort(_regime_score(centroids), kind="mergesort")
    return {
        int(order[0]): REGIME_RISK_OFF,
        int(order[1]): REGIME_SIDEWAYS,
        int(order[2]): REGIME_TREND_UP,
    }


def train_regime_kmeans_v4_pit(
    features: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    lookback_years: int = 3,
) -> RegimeKMeansV4Model:
    as_of = pd.Timestamp(as_of_date)
    start = as_of - pd.DateOffset(years=lookback_years)

    work = features.copy()
    work["as_of_date"] = pd.to_datetime(work["as_of_date"], errors="coerce")
    work = work[(work["as_of_date"] <= as_of) & (work["as_of_date"] > start)]
    work = _normalize_features(work, feature_columns)

    if len(work) < 12:
        dim = len(feature_columns)
        return RegimeKMeansV4Model(
            feature_columns=tuple(feature_columns),
            centroids=np.zeros((3, dim), dtype=float),
            means=np.zeros(dim, dtype=float),
            stds=np.ones(dim, dtype=float),
            label_by_cluster={0: REGIME_SIDEWAYS, 1: REGIME_SIDEWAYS, 2: REGIME_SIDEWAYS},
            trained_on_end=as_of,
            trained_rows=len(work),
            is_fallback=True,
        )

    x = work[list(feature_columns)].to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = np.where(x.std(axis=0) <= 1e-12, 1.0, x.std(axis=0))
    z = (x - means) / stds
    centroids = _simple_kmeans(z, k=3)

    return RegimeKMeansV4Model(
        feature_columns=tuple(feature_columns),
        centroids=centroids,
        means=means,
        stds=stds,
        label_by_cluster=_stable_label_map(centroids),
        trained_on_end=as_of,
        trained_rows=len(work),
        is_fallback=False,
    )


def predict_regime_kmeans_v4(model: RegimeKMeansV4Model, features: pd.DataFrame) -> pd.Series:
    work = _normalize_features(features, model.feature_columns)
    if work.empty:
        return pd.Series(dtype=object)

    x = work[list(model.feature_columns)].to_numpy(dtype=float)
    z = (x - model.means) / np.where(model.stds <= 1e-12, 1.0, model.stds)
    distances = np.linalg.norm(z[:, None, :] - model.centroids[None, :, :], axis=2)
    clusters = np.argmin(distances, axis=1)
    labels = [model.label_by_cluster[int(c)] for c in clusters]
    return pd.Series(labels, index=work.index, dtype=object)


def monitor_regime_feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    psi_threshold: float = 0.25,
) -> dict[str, object]:
    baseline_n = _normalize_features(baseline, feature_columns)
    current_n = _normalize_features(current, feature_columns)
    metrics = {col: compute_psi(current=current_n[col], baseline=baseline_n[col]) for col in feature_columns}

    breached_features = sorted([name for name, value in metrics.items() if value > psi_threshold])
    return {
        "trained_features": list(feature_columns),
        "psi": metrics,
        "psi_threshold": float(psi_threshold),
        "breached_features": breached_features,
        "governance_warning": bool(breached_features),
    }


def monthly_pit_retrain_schedule(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    lookback_years: int = 3,
) -> list[RegimeKMeansV4Model]:
    work = features.copy()
    work["as_of_date"] = pd.to_datetime(work["as_of_date"], errors="coerce")
    work = work.dropna(subset=["as_of_date"]).sort_values("as_of_date")
    month_ends = pd.to_datetime(work["as_of_date"]).dt.to_period("M").drop_duplicates().dt.to_timestamp("M")

    return [
        train_regime_kmeans_v4_pit(
            work,
            as_of,
            feature_columns=feature_columns,
            lookback_years=lookback_years,
        )
        for as_of in month_ends
    ]

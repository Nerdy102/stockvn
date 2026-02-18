from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class PairwiseRankConfig:
    model_id: str = "alpha_rankpair_v1"
    c: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 2000
    pairs_per_date: int = 400
    cv_splits: int = 5
    embargo_days: int = 5
    purge_horizon_days: int = 21


@dataclass
class PairwiseRankModel:
    config: PairwiseRankConfig
    feature_columns: list[str]
    estimator: LogisticRegression


def _zscore(x: pd.Series) -> pd.Series:
    s = x.astype(float)
    std = float(s.std(ddof=0))
    if std <= 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - float(s.mean())) / std


def sample_pairs_for_date(
    x: np.ndarray,
    y_rank_z: np.ndarray,
    as_of_date: dt.date,
    m_pairs: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y_rank_z)
    if n < 2 or m_pairs <= 0:
        return np.empty((0, x.shape[1]), dtype=float), np.empty((0,), dtype=int)

    rng = np.random.default_rng(42 + int(as_of_date.toordinal()))
    i_idx = rng.integers(0, n, size=m_pairs)
    j_idx = rng.integers(0, n, size=m_pairs)
    same = i_idx == j_idx
    if same.any():
        j_idx[same] = (j_idx[same] + 1) % n

    x_diff = x[i_idx] - x[j_idx]
    y = (y_rank_z[i_idx] > y_rank_z[j_idx]).astype(int)
    return x_diff.astype(float), y.astype(int)


def build_pairwise_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "y_rank_z",
    date_col: str = "as_of_date",
    pairs_per_date: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []

    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    work = work.dropna(subset=[label_col])

    for day, g in work.groupby(date_col, sort=True):
        x_day = g[feature_columns].astype(float).to_numpy()
        y_day = g[label_col].astype(float).to_numpy()
        x_pair, y_pair = sample_pairs_for_date(x_day, y_day, day, m_pairs=pairs_per_date)
        if len(y_pair) == 0:
            continue
        x_parts.append(x_pair)
        y_parts.append(y_pair)
        d_parts.append(np.array([day] * len(y_pair), dtype=object))

    if not x_parts:
        return np.empty((0, len(feature_columns))), np.empty((0,), dtype=int), np.empty((0,), dtype=object)

    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(d_parts)


def purged_kfold_embargo_date_splits(
    dates: list[dt.date],
    n_splits: int = 5,
    purge_horizon_days: int = 21,
    embargo_days: int = 5,
) -> list[tuple[list[dt.date], list[dt.date]]]:
    unique_dates = sorted(set(dates))
    if not unique_dates:
        return []
    idx = np.arange(len(unique_dates))
    folds = [f.tolist() for f in np.array_split(idx, n_splits) if len(f) > 0]

    out: list[tuple[list[dt.date], list[dt.date]]] = []
    for test_idx in folds:
        test_set = set(test_idx)
        test_min = min(test_idx)
        test_max = max(test_idx)
        train_idx: list[int] = []
        for i in idx.tolist():
            if i in test_set:
                continue
            if (i + purge_horizon_days) >= test_min and i <= test_max:
                continue
            if test_max < i <= (test_max + embargo_days):
                continue
            train_idx.append(i)
        train_dates = [unique_dates[i] for i in train_idx]
        test_dates = [unique_dates[i] for i in test_idx]
        out.append((train_dates, test_dates))
    return out


def train_pairwise_ranker(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: PairwiseRankConfig | None = None,
) -> PairwiseRankModel | None:
    cfg = config or PairwiseRankConfig()
    if frame.empty:
        return None

    x_pair, y_pair, d_pair = build_pairwise_dataset(
        frame,
        feature_columns=feature_columns,
        label_col="y_rank_z",
        date_col="as_of_date",
        pairs_per_date=cfg.pairs_per_date,
    )
    if len(y_pair) == 0:
        return None

    splits = purged_kfold_embargo_date_splits(
        dates=[d for d in pd.to_datetime(frame["as_of_date"]).dt.date.tolist() if pd.notna(d)],
        n_splits=cfg.cv_splits,
        purge_horizon_days=cfg.purge_horizon_days,
        embargo_days=cfg.embargo_days,
    )
    for train_dates, test_dates in splits:
        train_mask = np.isin(d_pair, train_dates)
        test_mask = np.isin(d_pair, test_dates)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        y_train = y_pair[train_mask]
        if len(np.unique(y_train)) < 2:
            continue
        cv_model = LogisticRegression(
            C=cfg.c,
            penalty=cfg.penalty,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
        )
        cv_model.fit(x_pair[train_mask], y_train)
        _ = cv_model.decision_function(x_pair[test_mask])

    model = LogisticRegression(
        C=cfg.c,
        penalty=cfg.penalty,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
    )
    model.fit(x_pair, y_pair)
    return PairwiseRankModel(config=cfg, feature_columns=feature_columns, estimator=model)


def predict_rank_score(
    model: PairwiseRankModel,
    frame: pd.DataFrame,
    date_col: str = "as_of_date",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "raw_score", "score_z"])

    out = frame[["symbol", date_col]].copy().rename(columns={date_col: "as_of_date"})
    x = frame.reindex(columns=model.feature_columns, fill_value=0.0).astype(float)
    out["raw_score"] = model.estimator.decision_function(x.to_numpy())
    out["score_z"] = out.groupby("as_of_date", group_keys=False)["raw_score"].apply(_zscore)
    out["raw_score"] = pd.to_numeric(out["raw_score"], errors="coerce").fillna(0.0)
    out["score_z"] = pd.to_numeric(out["score_z"], errors="coerce").fillna(0.0)
    return out


def validate_purged_cv_no_leakage(
    splits: list[tuple[list[dt.date], list[dt.date]]],
    purge_horizon_days: int,
    embargo_days: int,
) -> bool:
    for train_dates, test_dates in splits:
        if not test_dates:
            continue
        tmin = min(test_dates)
        tmax = max(test_dates)
        for d in train_dates:
            if d in test_dates:
                return False
            dist_to_test_start = (tmin - d).days
            if 0 <= dist_to_test_start <= purge_horizon_days:
                return False
            dist_after_test_end = (d - tmax).days
            if 0 < dist_after_test_end <= embargo_days:
                return False
    return True

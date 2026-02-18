from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core.alpha_v3.features import FEATURE_VERSION
from core.alpha_v3.targets import LABEL_VERSION

_HAS_SK = importlib.util.find_spec("sklearn") is not None
if _HAS_SK:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

_HAS_JOBLIB = importlib.util.find_spec("joblib") is not None
if _HAS_JOBLIB:
    import joblib  # type: ignore[import-untyped]


@dataclass
class AlphaV3Config:
    model_id: str = "alpha_v3"
    ridge_alpha: float = 10.0
    hgb_params: dict[str, Any] | None = None
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ridge_alpha": self.ridge_alpha,
            "hgb_params": self.hgb_params or default_hgbr_params(self.random_seed),
            "random_seed": self.random_seed,
            "feature_version": FEATURE_VERSION,
            "label_version": LABEL_VERSION,
            "score_formula": "0.55*pred_base + 0.45*mu - 0.35*uncert",
            "pred_base_formula": "0.2*pred_ridge + 0.8*pred_hgbr_rank",
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _FallbackModel:
    def __init__(self, quantile: float | None = None) -> None:
        self.quantile = quantile
        self.constant_ = 0.0

    def fit(self, _x, y):
        arr = np.asarray(y, dtype=float)
        self.constant_ = (
            float(np.nanquantile(arr, self.quantile))
            if self.quantile is not None
            else float(np.nanmean(arr))
        )
        return self

    def predict(self, x):
        return np.full((len(x),), self.constant_, dtype=float)


def default_hgbr_params(seed: int) -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "max_iter": 400,
        "early_stopping": False,
        "random_state": seed,
    }


def ridge_rank_v3(alpha: float = 10.0):
    if _HAS_SK:
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha, fit_intercept=True))])
    return _FallbackModel()


def hgbr_rank_v3(hgb_params: dict[str, Any] | None = None):
    if _HAS_SK:
        return HistGradientBoostingRegressor(loss="squared_error", **(hgb_params or default_hgbr_params(42)))
    return _FallbackModel()


def hgbr_quantiles_v3(hgb_params: dict[str, Any] | None = None):
    if _HAS_SK:
        base = hgb_params or default_hgbr_params(42)
        return (
            HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **base),
            HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **base),
            HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **base),
        )
    return (_FallbackModel(0.1), _FallbackModel(0.5), _FallbackModel(0.9))


def _sanitize_xy(x, y=None):
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y is None:
        return x_arr
    y_arr = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return x_arr, y_arr


def _enforce_quantile_monotonicity(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(np.vstack([q10, q50, q90]), axis=0)
    return ordered[0], ordered[1], ordered[2]


def compose_alpha_v3_score(
    pred_ridge: np.ndarray,
    pred_hgbr_rank: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> dict[str, np.ndarray]:
    q10o, q50o, q90o = _enforce_quantile_monotonicity(q10, q50, q90)
    pred_base = 0.2 * pred_ridge + 0.8 * pred_hgbr_rank
    mu = q50o
    uncert = np.maximum(0.0, q90o - q10o)
    score = 0.55 * pred_base + 0.45 * mu - 0.35 * uncert
    return {
        "pred_base": pred_base,
        "mu": mu,
        "uncert": uncert,
        "score": score,
        "hgbr_q10_v3": q10o,
        "hgbr_q50_v3": q50o,
        "hgbr_q90_v3": q90o,
    }


class AlphaV3ModelBundle:
    def __init__(self, config: AlphaV3Config | None = None) -> None:
        self.config = config or AlphaV3Config()
        hgb_params = self.config.hgb_params or default_hgbr_params(self.config.random_seed)
        self.ridge = ridge_rank_v3(alpha=self.config.ridge_alpha)
        self.hgbr = hgbr_rank_v3(hgb_params=hgb_params)
        self.q10, self.q50, self.q90 = hgbr_quantiles_v3(hgb_params=hgb_params)

    def fit(self, x, y):
        x_arr, y_arr = _sanitize_xy(x, y)
        self.ridge.fit(x_arr, y_arr)
        self.hgbr.fit(x_arr, y_arr)
        self.q10.fit(x_arr, y_arr)
        self.q50.fit(x_arr, y_arr)
        self.q90.fit(x_arr, y_arr)
        return self

    def predict_components(self, x) -> dict[str, np.ndarray]:
        x_arr = _sanitize_xy(x)
        pred_ridge = np.asarray(self.ridge.predict(x_arr), dtype=float)
        pred_hgbr_rank = np.asarray(self.hgbr.predict(x_arr), dtype=float)
        pred_q10 = np.asarray(self.q10.predict(x_arr), dtype=float)
        pred_q50 = np.asarray(self.q50.predict(x_arr), dtype=float)
        pred_q90 = np.asarray(self.q90.predict(x_arr), dtype=float)
        comp = compose_alpha_v3_score(pred_ridge, pred_hgbr_rank, pred_q10, pred_q50, pred_q90)
        comp["ridge_rank_v3"] = pred_ridge
        comp["hgbr_rank_v3"] = pred_hgbr_rank
        return comp

    def dump(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_JOBLIB:
            joblib.dump(self.ridge, output_dir / "ridge.joblib")
            joblib.dump(self.hgbr, output_dir / "hgbr.joblib")
            joblib.dump(self.q10, output_dir / "q10.joblib")
            joblib.dump(self.q50, output_dir / "q50.joblib")
            joblib.dump(self.q90, output_dir / "q90.joblib")
            return
        for name in ["ridge", "hgbr", "q10", "q50", "q90"]:
            (output_dir / f"{name}.joblib").write_text("joblib-not-installed", encoding="utf-8")


def load_alpha_v3_bundle(artifact_dir: Path) -> AlphaV3ModelBundle:
    if not _HAS_JOBLIB:
        raise RuntimeError("joblib is required to load persisted alpha_v3 models")
    bundle = AlphaV3ModelBundle()
    bundle.ridge = joblib.load(artifact_dir / "ridge.joblib")
    bundle.hgbr = joblib.load(artifact_dir / "hgbr.joblib")
    bundle.q10 = joblib.load(artifact_dir / "q10.joblib")
    bundle.q50 = joblib.load(artifact_dir / "q50.joblib")
    bundle.q90 = joblib.load(artifact_dir / "q90.joblib")
    return bundle


def metadata_payload(
    train_start,
    train_end,
    git_commit: str,
    config: AlphaV3Config,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "train_start": str(train_start),
        "train_end": str(train_end),
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "git_commit": git_commit,
        "config_hash": config.config_hash(),
    }
    if feature_columns is not None:
        payload["feature_columns"] = feature_columns
    return payload


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_matrix_from_records(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = {
        "symbol",
        "date",
        "as_of_date",
        "y_excess",
        "y_rank_z",
        "label_version",
        "feature_version",
    }
    cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols], cols


def align_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = df.reindex(columns=feature_columns, fill_value=0.0)
    for c in feature_columns:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    return aligned.fillna(0.0)

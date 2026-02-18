from __future__ import annotations

import importlib.util

import numpy as np

_HAS_SK = importlib.util.find_spec("sklearn") is not None
if _HAS_SK:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


def _hgb_params() -> dict:
    return {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "max_iter": 400,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
    }


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


def ridge_rank_v2():
    if _HAS_SK:
        return Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, fit_intercept=True))]
        )
    return _FallbackModel()


def hgbr_rank_v2():
    if _HAS_SK:
        return HistGradientBoostingRegressor(loss="squared_error", **_hgb_params())
    return _FallbackModel()


def hgbr_quantiles_v2() -> tuple[object, object, object]:
    if _HAS_SK:
        return (
            HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **_hgb_params()),
            HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **_hgb_params()),
            HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **_hgb_params()),
        )
    return (_FallbackModel(0.1), _FallbackModel(0.5), _FallbackModel(0.9))


def _sanitize_xy(x, y=None):
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y is None:
        return x_arr
    y_arr = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return x_arr, y_arr


class MlModelV2Bundle:
    def __init__(self) -> None:
        self.ridge = ridge_rank_v2()
        self.hgbr = hgbr_rank_v2()
        self.q10, self.q50, self.q90 = hgbr_quantiles_v2()

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
        pred_q10 = np.asarray(self.q10.predict(x_arr), dtype=float)
        pred_q50 = np.asarray(self.q50.predict(x_arr), dtype=float)
        pred_q90 = np.asarray(self.q90.predict(x_arr), dtype=float)
        ordered = np.sort(np.vstack([pred_q10, pred_q50, pred_q90]), axis=0)
        pred_q10, pred_q50, pred_q90 = ordered
        mu = pred_q50
        uncert = pred_q90 - pred_q10
        score_final = mu - 0.35 * uncert
        score_rank_z = np.asarray(self.hgbr.predict(x_arr), dtype=float)
        return {
            "ridge_rank_v2": np.asarray(self.ridge.predict(x_arr), dtype=float),
            "hgbr_rank_v2": score_rank_z,
            "hgbr_q10_v2": pred_q10,
            "hgbr_q50_v2": pred_q50,
            "hgbr_q90_v2": pred_q90,
            "mu": mu,
            "uncert": uncert,
            "score_final": score_final,
            "score_rank_z": score_rank_z,
        }

from __future__ import annotations

import importlib.util

import numpy as np

_HAS_SK = importlib.util.find_spec("sklearn") is not None
if _HAS_SK:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


class _FallbackRidge:
    def fit(self, x, y):
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        x1 = np.c_[np.ones(len(x)), x]
        a = x1.T @ x1 + 10.0 * np.eye(x1.shape[1])
        b = x1.T @ y
        self.beta = np.linalg.lstsq(a, b, rcond=None)[0]
        return self

    def predict(self, x):
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        x1 = np.c_[np.ones(len(x)), x]
        return x1 @ self.beta


class _FallbackHGBR:
    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        self.mean = float(np.nanmean(y))
        return self

    def predict(self, x):
        return np.full(shape=(len(x),), fill_value=self.mean)


def _sanitize_xy(x, y=None):
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y is None:
        return x_arr
    y_arr = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return x_arr, y_arr


class MlModelBundle:
    def __init__(self) -> None:
        if _HAS_SK:
            self.ridge = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0, fit_intercept=True)),
                ]
            )
            self.hgbr = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=6,
                max_leaf_nodes=31,
                min_samples_leaf=200,
                l2_regularization=1.0,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        else:
            self.ridge = _FallbackRidge()
            self.hgbr = _FallbackHGBR()

    def fit(self, x, y):
        x_arr, y_arr = _sanitize_xy(x, y)
        self.ridge.fit(x_arr, y_arr)
        self.hgbr.fit(x_arr, y_arr)
        return self

    def predict(self, x):
        x_arr = _sanitize_xy(x)
        y1 = self.ridge.predict(x_arr)
        y2 = self.hgbr.predict(x_arr)
        return 0.2 * np.asarray(y1) + 0.8 * np.asarray(y2)

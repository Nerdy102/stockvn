from __future__ import annotations

import importlib.util

import numpy as np

_HAS_SK = importlib.util.find_spec("sklearn") is not None
if _HAS_SK:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


class _FallbackMean:
    def __init__(self, quantile: float | None = None) -> None:
        self.quantile = quantile
        self.val = 0.0

    def fit(self, _x, y):
        arr = np.asarray(y, dtype=float)
        if self.quantile is None:
            self.val = float(np.nanmean(arr))
        else:
            self.val = float(np.nanquantile(arr, self.quantile))
        return self

    def predict(self, x):
        return np.full(shape=(len(x),), fill_value=self.val)


class MlModelV2Bundle:
    def __init__(self) -> None:
        if _HAS_SK:
            self.ridge_rank_v2 = Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, fit_intercept=True))]
            )
            self.hgbr_rank_v2 = HistGradientBoostingRegressor(
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

            def _q(q: float):
                return HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
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

            self.hgbr_q10_v2 = _q(0.1)
            self.hgbr_q50_v2 = _q(0.5)
            self.hgbr_q90_v2 = _q(0.9)
        else:
            self.ridge_rank_v2 = _FallbackMean(None)
            self.hgbr_rank_v2 = _FallbackMean(None)
            self.hgbr_q10_v2 = _FallbackMean(0.1)
            self.hgbr_q50_v2 = _FallbackMean(0.5)
            self.hgbr_q90_v2 = _FallbackMean(0.9)

    def fit(self, x, y):
        self.ridge_rank_v2.fit(x, y)
        self.hgbr_rank_v2.fit(x, y)
        self.hgbr_q10_v2.fit(x, y)
        self.hgbr_q50_v2.fit(x, y)
        self.hgbr_q90_v2.fit(x, y)
        return self

    def predict_components(self, x) -> dict[str, np.ndarray]:
        q10 = np.asarray(self.hgbr_q10_v2.predict(x), dtype=float)
        q50 = np.asarray(self.hgbr_q50_v2.predict(x), dtype=float)
        q90 = np.asarray(self.hgbr_q90_v2.predict(x), dtype=float)
        ordered = np.sort(np.vstack([q10, q50, q90]), axis=0)
        q10, q50, q90 = ordered[0], ordered[1], ordered[2]
        mu = q50
        uncert = q90 - q10
        score_final = mu - 0.35 * uncert
        rank = np.asarray(self.hgbr_rank_v2.predict(x), dtype=float)
        return {
            "ridge_rank_v2": np.asarray(self.ridge_rank_v2.predict(x), dtype=float),
            "hgbr_rank_v2": rank,
            "hgbr_q10_v2": q10,
            "hgbr_q50_v2": q50,
            "hgbr_q90_v2": q90,
            "mu": mu,
            "uncert": uncert,
            "score_rank_z": rank,
            "score_final": score_final,
        }

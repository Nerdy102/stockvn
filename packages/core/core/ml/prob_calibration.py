from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from core.alpha_v3.calibration import compute_probability_calibration_metrics

_HAS_SK = True
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _HAS_SK = False


@dataclass
class CalibratedProbabilityModel:
    a: float
    b: float
    iso_x: np.ndarray
    iso_y: np.ndarray

    def p_raw(self, score_z: np.ndarray) -> np.ndarray:
        s = np.asarray(score_z, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.a * s + self.b)))

    def p_cal(self, score_z: np.ndarray) -> np.ndarray:
        p = self.p_raw(score_z)
        if self.iso_x.size == 0:
            return np.clip(p, 0.0, 1.0)
        return np.clip(np.interp(p, self.iso_x, self.iso_y, left=float(self.iso_y[0]), right=float(self.iso_y[-1])), 0.0, 1.0)


def fit_logistic_1d(score_z: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    x = np.asarray(score_z, dtype=float).reshape(-1, 1)
    y = np.asarray(z, dtype=float)
    y = (y > 0.5).astype(int)
    if x.shape[0] == 0:
        return 1.0, 0.0
    if np.unique(y).size < 2:
        return 1.0, float(np.log(np.clip(np.mean(y), 1e-6, 1 - 1e-6) / np.clip(1 - np.mean(y), 1e-6, 1)))

    if _HAS_SK:
        lr = LogisticRegression(random_state=42, solver="lbfgs")
        lr.fit(x, y)
        return float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])

    # Fallback simple GD
    a, b = 0.0, 0.0
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(a * x.ravel() + b)))
        da = float(np.mean((p - y) * x.ravel()))
        db = float(np.mean(p - y))
        a -= 0.1 * da
        b -= 0.1 * db
    return float(a), float(b)


def fit_isotonic_validation_only(p_raw_val: np.ndarray, z_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_raw_val, dtype=float)
    z = (np.asarray(z_val, dtype=float) > 0.5).astype(float)
    if p.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if _HAS_SK:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, z)
        x_thr = np.asarray(getattr(iso, "X_thresholds_", []), dtype=float)
        y_thr = np.asarray(getattr(iso, "y_thresholds_", []), dtype=float)
        if x_thr.size > 0 and y_thr.size > 0:
            return x_thr, y_thr

    order = np.argsort(p)
    p_sorted = p[order]
    z_sorted = z[order]
    csum = np.cumsum(z_sorted)
    y_hat = csum / np.arange(1, len(z_sorted) + 1)
    return p_sorted, y_hat


def fit_calibrated_probability_model(
    train_score_z: np.ndarray,
    train_z: np.ndarray,
    val_score_z: np.ndarray,
    val_z: np.ndarray,
) -> CalibratedProbabilityModel:
    a, b = fit_logistic_1d(train_score_z, train_z)
    p_val_raw = 1.0 / (1.0 + np.exp(-(a * np.asarray(val_score_z, dtype=float) + b)))
    iso_x, iso_y = fit_isotonic_validation_only(p_val_raw, val_z)
    return CalibratedProbabilityModel(a=a, b=b, iso_x=iso_x, iso_y=iso_y)


def calibration_metrics(p: np.ndarray, z: np.ndarray, bins: int = 10) -> dict:
    return compute_probability_calibration_metrics(p.tolist(), z.tolist(), bins=bins)

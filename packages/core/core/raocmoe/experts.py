from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.ml.models import MlModelBundle


class ExpertSet:
    def __init__(self) -> None:
        self.bundle = MlModelBundle()
        self.factor_cols: list[str] = []
        self.factor_ready = False
        self.residuals: deque[float] = deque(maxlen=256)

    def fit_factor_model(self, x: np.ndarray, y: np.ndarray, cols: list[str]) -> None:
        if len(x) < 32:
            return
        self.bundle.fit(x, y)
        self.factor_cols = list(cols)
        preds = self.bundle.predict(x)
        for r in y - preds:
            self.residuals.append(float(r))
        self.factor_ready = True

    def factor_ml(self, feature_row: dict[str, float]) -> tuple[float, float]:
        if not self.factor_ready or not self.factor_cols:
            return 0.0, 0.1
        x = np.array([[float(feature_row.get(c, 0.0)) for c in self.factor_cols]], dtype=float)
        mu = float(self.bundle.predict(x)[0])
        resid_std = (
            float(np.std(np.asarray(self.residuals, dtype=float))) if self.residuals else 1.0
        )
        conf = 1.0 / max(1e-6, resid_std)
        return mu, conf

    def trend_tech(
        self, price: float, ema20: float, ema50: float, k_trend: float = 0.6
    ) -> tuple[float, float]:
        strength = math.tanh((ema20 - ema50) / max(price, 1e-8))
        mu = max(-0.1, min(0.1, strength * k_trend))
        return float(mu), float(abs(strength))

    def meanrev_vwap(
        self, price: float, vwap: float, liq_stress: float, k_mr: float = 0.8
    ) -> tuple[float, float]:
        dev = (price - vwap) / max(1e-8, price)
        mu = max(-0.1, min(0.1, -dev * k_mr))
        conf = abs(dev) * max(0.0, 1.0 - liq_stress)
        return float(mu), float(conf)

    def defensive_lowvol(self, regime: str) -> tuple[float, float]:
        if regime == "PANIC_VOL":
            return -0.02, 1.0
        if regime == "RISK_OFF":
            return -0.005, 0.6
        return 0.003, 0.2

from __future__ import annotations

import math

from .types import MoEOutput


class HedgeRouter:
    def __init__(
        self, regimes: list[str], experts: list[str], eta: float, loss_clip: float
    ) -> None:
        self.eta = float(eta)
        self.loss_clip = float(loss_clip)
        self.min_weight = 0.02
        self.weights = {r: {e: 1.0 / len(experts) for e in experts} for r in regimes}

    def combine(self, regime: str, expert_mu: dict[str, float]) -> MoEOutput:
        w = self.weights[regime]
        mu = float(sum(w[k] * float(expert_mu[k]) for k in expert_mu))
        ent = -sum(v * math.log(max(v, 1e-12)) for v in w.values())
        return MoEOutput(
            mu_hat_combined=mu, expert_mus=dict(expert_mu), weights=dict(w), entropy=float(ent)
        )

    def update(
        self,
        regime: str,
        expert_mu: dict[str, float],
        realized_r: float,
        realized_vol: float | None = None,
        vol: float | None = None,
    ) -> None:
        w = self.weights[regime]
        base_vol = realized_vol if realized_vol is not None else vol
        eps = max(1e-8, float(base_vol if base_vol is not None else 1.0))
        for k, mu in expert_mu.items():
            sgn = 1.0 if mu >= 0.0 else -1.0
            loss = -sgn * float(realized_r) / eps
            loss = max(-self.loss_clip, min(self.loss_clip, loss))
            w[k] = max(self.min_weight, w[k] * math.exp(-self.eta * loss))
        total = sum(w.values())
        self.weights[regime] = {k: v / total for k, v in w.items()}

from __future__ import annotations

import math
from enum import Enum

from .types import CPEvent, RegimePosterior


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    SIDEWAYS = "SIDEWAYS"
    RISK_OFF = "RISK_OFF"
    PANIC_VOL = "PANIC_VOL"


class RegimeEngine:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["regime"]
        self.states = [Regime(x).value for x in self.cfg["states"]]
        p0 = 1.0 / max(1, len(self.states))
        self.posterior = {z: p0 for z in self.states}
        self.hard_regime = self.states[0]
        self.last_switch_t = 0

    def _emission_logits(self, features: dict[str, float], cp_bias: float) -> dict[str, float]:
        out: dict[str, float] = {}
        fw = self.cfg["feature_weights"]
        for z in self.states:
            score = 0.0
            for key, wt in fw[z].items():
                score += float(wt) * float(features.get(key, 0.0))
            if z in {Regime.RISK_OFF.value, Regime.PANIC_VOL.value}:
                score += cp_bias
            out[z] = score
        return out

    def _cp_bias(self, cp_events: list[CPEvent]) -> float:
        if not cp_events:
            return 0.0
        bump = 0.0
        for ev in cp_events:
            if ev.detector_name in {"vol_shift", "liq_shift"}:
                bump += min(2.0, ev.score / 8.0)
        return bump

    def update(
        self,
        t: int,
        features: dict[str, float],
        cp_events: list[CPEvent] | None = None,
    ) -> RegimePosterior:
        cp_bias = self._cp_bias(cp_events or [])
        trans = self.cfg["transition_matrix"]
        prior = {
            z: sum(self.posterior[p] * float(trans[p][z]) for p in self.states) for z in self.states
        }
        logits = self._emission_logits(features, cp_bias)
        post_logits = {z: math.log(max(prior[z], 1e-12)) + logits[z] for z in self.states}
        mx = max(post_logits.values())
        expv = {z: math.exp(v - mx) for z, v in post_logits.items()}
        den = sum(expv.values())
        self.posterior = {z: expv[z] / den for z in self.states}

        hys = self.cfg["hysteresis"]
        p_on = float(hys["p_on"])
        p_off = float(hys["p_off"])
        cooldown = int(hys["cooldown_bars"])
        best = max(self.posterior, key=self.posterior.get)
        since = t - self.last_switch_t
        if since >= cooldown:
            if best != self.hard_regime and self.posterior[best] > p_on:
                if self.posterior[self.hard_regime] < p_off:
                    self.hard_regime = best
                    self.last_switch_t = t

        return RegimePosterior(
            posterior={k: float(v) for k, v in self.posterior.items()},
            hard_regime=self.hard_regime,
            time_since_last_switch=t - self.last_switch_t,
            debug={"best": best, "cp_bias": cp_bias, "since": since},
        )

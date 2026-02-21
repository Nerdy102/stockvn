from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .types import UQInterval


@dataclass
class _PendingPred:
    t: int
    regime: str
    symbol: str
    horizon: str
    yhat: float
    interval_low: float
    interval_high: float
    scale: float


class UncertaintyEngine:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["uq"]
        self.horizons = list(cfg["alpha_target_horizons"])
        self.horizon_steps = {str(h["name"]): int(h["steps"]) for h in self.horizons}
        self.target_alpha = {str(h["name"]): float(h["target_miscoverage"]) for h in self.horizons}
        self.max_scores = int(self.cfg["regime_calibrators"]["max_scores_per_pool"])
        self.min_pool_size = int(self.cfg["pooling"]["min_pool_size"])
        self.roll_window = int(self.cfg["coverage_monitor"]["rolling_window"])
        self.tolerance = float(self.cfg["coverage_monitor"]["undercoverage_tolerance"])
        self.pause_breaches = int(self.cfg["coverage_monitor"]["consecutive_breaches_to_pause"])

        self.scores = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_scores)))
        )
        self.sector_scores = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_scores)))
        )
        self.alphas = defaultdict(dict)
        self.coverage = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.roll_window)))
        self.consecutive_breaches = defaultdict(lambda: defaultdict(int))
        self.pending = defaultdict(deque)

    def _quantile(self, vals: list[float], q: float) -> float:
        if not vals:
            return 1.0
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
        return float(s[idx])

    def _pool_scores(self, regime: str, symbol: str, sector: str, horizon: str) -> list[float]:
        vals = list(self.scores[regime][symbol][horizon])
        sec = list(self.sector_scores[regime][sector][horizon]) if sector else []
        if len(sec) >= self.min_pool_size:
            q_symbol = self._quantile(vals, 0.9)
            q_sector = self._quantile(sec, 0.9)
            use = sec if q_sector >= q_symbol else vals
            return list(use)
        return vals

    def get_intervals(
        self,
        t: int,
        regime: str,
        symbol: str,
        sector: str,
        yhat: float,
        realized_vol: float,
    ) -> tuple[list[UQInterval], float]:
        scale = max(float(realized_vol), 1e-8)
        out: list[UQInterval] = []
        widths: list[float] = []
        for h in self.horizons:
            name = str(h["name"])
            alpha = float(self.alphas[regime].get(name, self.target_alpha[name]))
            pool = self._pool_scores(regime, symbol, sector, name)
            qscore = self._quantile(pool, 1.0 - alpha) if pool else 1.64
            width = qscore * scale
            out.append(
                UQInterval(horizon=name, lower=yhat - width, upper=yhat + width, alpha=alpha)
            )
            widths.append(width)
            self.pending[name].append(
                _PendingPred(
                    t=t,
                    regime=regime,
                    symbol=symbol,
                    horizon=name,
                    yhat=yhat,
                    interval_low=yhat - width,
                    interval_high=yhat + width,
                    scale=scale,
                )
            )
        return out, float(sum(widths) / max(1, len(widths)))

    def _gamma(self, psi: float, cp_score: float) -> float:
        aci = self.cfg["aci"]
        gamma = float(aci["gamma_base"])
        if bool(aci["adaptive_gamma"]["enabled"]):
            gamma *= 1.0 + float(aci["adaptive_gamma"]["psi_weight"]) * max(0.0, psi)
            gamma *= 1.0 + float(aci["adaptive_gamma"]["cpd_weight"]) * max(0.0, cp_score)
        return min(float(aci["gamma_max"]), max(float(aci["gamma_min"]), gamma))

    def update_with_label(
        self,
        t: int,
        horizon: str,
        y_true: float,
        psi: float,
        cp_score: float,
        sector: str,
    ) -> None:
        steps = self.horizon_steps[horizon]
        q = self.pending[horizon]
        while q and (t - q[0].t) >= steps:
            pred = q.popleft()
            err = int(not (pred.interval_low <= y_true <= pred.interval_high))
            score = abs(y_true - pred.yhat) / pred.scale
            self.scores[pred.regime][pred.symbol][horizon].append(float(score))
            if sector:
                self.sector_scores[pred.regime][sector][horizon].append(float(score))
            self.coverage[pred.regime][horizon].append(1 - err)
            gamma = self._gamma(psi=psi, cp_score=cp_score)
            alpha = float(self.alphas[pred.regime].get(horizon, self.target_alpha[horizon]))
            alpha = alpha + gamma * (self.target_alpha[horizon] - err)
            self.alphas[pred.regime][horizon] = min(0.4, max(0.01, alpha))

    def warm_start_regime(self, new_regime: str, old_regime: str) -> None:
        ws = int(self.cfg["regime_calibrators"]["warm_start_scores"])
        for symbol, by_h in self.scores[old_regime].items():
            for horizon, values in by_h.items():
                tail = list(values)[-ws:]
                self.scores[new_regime][symbol][horizon].extend(tail)

    def coverage_status(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for regime, by_h in self.coverage.items():
            out[regime] = {}
            for horizon, vals in by_h.items():
                if vals:
                    out[regime][horizon] = float(sum(vals) / len(vals))
        return out

    def governance_undercoverage(self) -> tuple[bool, dict[str, int]]:
        paused = False
        details: dict[str, int] = {}
        for regime, by_h in self.coverage.items():
            for horizon, vals in by_h.items():
                if len(vals) < self.roll_window:
                    continue
                miss = 1.0 - (sum(vals) / len(vals))
                breach = miss > (self.target_alpha[horizon] + self.tolerance)
                key = f"{regime}:{horizon}"
                if breach:
                    self.consecutive_breaches[regime][horizon] += 1
                else:
                    self.consecutive_breaches[regime][horizon] = 0
                details[key] = self.consecutive_breaches[regime][horizon]
                paused = paused or self.consecutive_breaches[regime][horizon] >= self.pause_breaches
        return paused, details

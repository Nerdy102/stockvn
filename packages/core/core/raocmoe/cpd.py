from __future__ import annotations

import math
from collections import defaultdict, deque

from .types import CPEvent, CPScore


class DynamicGeometricGrid:
    def __init__(self, geometric_base: int = 2, max_candidates: int = 64) -> None:
        self.base = max(2, int(geometric_base))
        self.max_candidates = max(8, int(max_candidates))
        self.candidates: list[int] = []

    def update(self, t: int) -> list[int]:
        self.candidates.append(t)
        by_bucket: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=2))
        for tau in self.candidates:
            age = max(1, t - tau + 1)
            bid = int(math.log(age, self.base))
            by_bucket[bid].append(tau)
        compact: list[int] = []
        for bid in sorted(by_bucket):
            compact.extend(list(by_bucket[bid]))
        self.candidates = sorted(set(compact))[-self.max_candidates :]
        return self.candidates


class CPDDetector:
    def __init__(
        self,
        name: str,
        robust_clip: float,
        threshold_base: float,
        cooldown_bars: int,
        geometric_base: int,
        max_candidates: int,
    ) -> None:
        self.name = name
        self.robust_clip = float(robust_clip)
        self.threshold_base = float(threshold_base)
        self.cooldown_bars = int(cooldown_bars)
        self.grid = DynamicGeometricGrid(
            geometric_base=geometric_base, max_candidates=max_candidates
        )
        self._prefix_x = [0.0]
        self._prefix_x2 = [0.0]
        self.last_cp_t = -10_000

    def _transform(self, y_t: float) -> float:
        return max(-self.robust_clip, min(self.robust_clip, float(y_t)))

    def _segment_stats(self, start: int, end: int) -> tuple[float, float]:
        n = max(1, end - start + 1)
        sx = self._prefix_x[end] - self._prefix_x[start - 1]
        sx2 = self._prefix_x2[end] - self._prefix_x2[start - 1]
        mean = sx / n
        var = max(0.0, sx2 / n - mean * mean)
        return mean, var

    def _score_segment(self, t: int, tau: int) -> float:
        mu_recent, _ = self._segment_stats(tau, t)
        mu_hist, _ = self._segment_stats(1, tau - 1)
        n = max(1, t - tau + 1)
        return abs(mu_recent - mu_hist) * math.sqrt(n)

    def update(self, y_t: float, t: int) -> tuple[bool, float, int | None]:
        x = self._transform(y_t)
        self._prefix_x.append(self._prefix_x[-1] + x)
        self._prefix_x2.append(self._prefix_x2[-1] + x * x)
        candidates = self.grid.update(t)

        best_score = 0.0
        tau_hat: int | None = None
        for tau in candidates:
            if tau <= 2:
                continue
            score = self._score_segment(t=t, tau=tau)
            if score > best_score:
                best_score = score
                tau_hat = tau

        threshold = self.threshold_base * math.log(1 + t) / math.log(2)
        in_cooldown = (t - self.last_cp_t) < self.cooldown_bars
        cp_flag = bool(best_score > threshold and not in_cooldown)
        if cp_flag:
            self.last_cp_t = t
        return cp_flag, float(best_score), tau_hat


class MeanShiftDetector(CPDDetector):
    pass


class VolShiftDetector(CPDDetector):
    def _transform(self, y_t: float) -> float:
        base = max(1e-8, abs(float(y_t)))
        return super()._transform(math.log(base))


class CorrShiftDetector(CPDDetector):
    pass


class LiquidityShiftDetector(CPDDetector):
    pass


def build_default_detectors(cfg: dict) -> dict[str, CPDDetector]:
    dcfg = cfg["cpd"]["detectors"]
    grid = cfg["cpd"]["grid"]
    cooldown = int(cfg["cpd"]["cooldown_bars_after_cp"])

    def _mk(cls, k: str) -> CPDDetector:
        return cls(
            name=k,
            robust_clip=float(dcfg[k]["robust_clip"]),
            threshold_base=float(dcfg[k]["threshold_base"]),
            cooldown_bars=cooldown,
            geometric_base=int(grid["geometric_base"]),
            max_candidates=int(grid["max_candidates"]),
        )

    return {
        "mean_shift": _mk(MeanShiftDetector, "mean_shift"),
        "vol_shift": _mk(VolShiftDetector, "vol_shift"),
        "corr_shift": _mk(CorrShiftDetector, "corr_shift"),
        "liq_shift": _mk(LiquidityShiftDetector, "liq_shift"),
    }


def update_all_detectors(
    detectors: dict[str, CPDDetector], values: dict[str, float], t: int
) -> tuple[list[CPEvent], dict[str, float]]:
    events: list[CPEvent] = []
    vector: dict[str, float] = {}
    for name, detector in detectors.items():
        cp_flag, score, tau_hat = detector.update(float(values.get(name, 0.0)), t)
        vector[name] = score
        if cp_flag:
            events.append(
                CPEvent(
                    detector_name=name,
                    tau_hat=tau_hat,
                    score=score,
                    timestamp=t,
                    cp_scores=[CPScore(detector=name, score=score)],
                )
            )
    return events, vector

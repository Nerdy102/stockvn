from __future__ import annotations

import math

from .types import PortfolioTarget


class PortfolioController:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["portfolio"]
        self.mu_clip = float(cfg["moe"]["outputs"]["mu_clip"])

    def _project_turnover(
        self, weights: dict[str, float], prev: dict[str, float], cap: float
    ) -> tuple[dict[str, float], float]:
        delta = {k: weights.get(k, 0.0) - prev.get(k, 0.0) for k in set(weights) | set(prev)}
        l1 = sum(abs(v) for v in delta.values())
        if l1 <= cap or l1 <= 1e-12:
            return weights, l1
        scale = cap / l1
        out = {k: prev.get(k, 0.0) + delta[k] * scale for k in delta}
        return out, cap

    def build_target(
        self,
        universe: list[str],
        mu: dict[str, float],
        uncertainty: dict[str, float],
        prev: dict[str, float],
        vols: dict[str, float],
        adv20: dict[str, float],
        nav: float,
        regime: str,
        sectors: dict[str, str] | None = None,
        paused: bool = False,
        participation_limit_override: float | None = None,
    ) -> PortfolioTarget:
        cash_min = float(self.cfg["target_cash_min"])
        if paused or regime == "PANIC_VOL":
            return PortfolioTarget(
                weights={s: 0.0 for s in universe},
                cash_weight=1.0,
                turnover=0.0,
                expected_cost_bps=0.0,
                debug={"paused": paused, "regime": regime},
            )

        lam = float(self.cfg["robust"]["uncertainty_penalty_lambda"])
        score: dict[str, float] = {}
        for sym in universe:
            s = max(-self.mu_clip, min(self.mu_clip, float(mu.get(sym, 0.0)))) - lam * float(
                uncertainty.get(sym, 0.0)
            )
            if bool(self.cfg["robust"]["worst_case"]):
                s -= float(uncertainty.get(sym, 0.0))
            score[sym] = s

        top = [
            k
            for k, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)[
                : min(30, len(universe))
            ]
        ]
        positive = [s for s in top if score[s] > 0.0]
        if not positive:
            return PortfolioTarget(
                weights={s: 0.0 for s in universe},
                cash_weight=1.0,
                turnover=sum(abs(prev.get(s, 0.0)) for s in universe),
                expected_cost_bps=0.0,
                debug={"all_cash": True},
            )

        raw = {s: score[s] / max(1e-8, float(vols.get(s, 0.02))) for s in positive}
        raw_sum = sum(max(0.0, x) for x in raw.values())
        invest_cap = 1.0 - cash_min
        w = {s: (max(0.0, raw.get(s, 0.0)) / raw_sum) * invest_cap for s in universe}

        max_single = float(self.cfg["max_single_weight"])
        w = {s: min(max_single, w[s]) for s in universe}

        sec_map = sectors or {s: "OTHER" for s in universe}
        sec_cap = float(self.cfg["max_sector_weight"])
        sector_totals: dict[str, float] = {}
        for s in universe:
            sec = sec_map.get(s, "OTHER")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w[s]
        for sec, total in sector_totals.items():
            if total > sec_cap and total > 0:
                ratio = sec_cap / total
                for s in universe:
                    if sec_map.get(s, "OTHER") == sec:
                        w[s] *= ratio

        participation = float(
            participation_limit_override or self.cfg["liquidity"]["participation_limit"]
        )
        days = float(self.cfg["liquidity"]["days_to_exit"])
        for s in universe:
            max_position = float(adv20.get(s, nav * 1000.0)) * participation * days
            w[s] = min(w[s], max_position / max(nav, 1e-8))

        w, turnover = self._project_turnover(
            weights=w,
            prev=prev,
            cap=float(self.cfg["turnover_cap_l1"]),
        )

        band = float(self.cfg["no_trade_band_abs"])
        for s in universe:
            if abs(w.get(s, 0.0) - prev.get(s, 0.0)) < band:
                w[s] = prev.get(s, 0.0)

        vol_est = math.sqrt(
            252.0 * sum((w.get(s, 0.0) * float(vols.get(s, 0.02))) ** 2 for s in universe)
        )
        vt = float(self.cfg["risk"]["vol_target_annual"])
        scale = min(1.0, vt / max(vol_est, 1e-8))
        if regime == "RISK_OFF":
            scale *= 0.5
        w = {s: max(0.0, w.get(s, 0.0) * scale) for s in universe}

        risky = sum(w.values())
        cash = max(cash_min, 1.0 - risky)
        exp_cost = turnover * 5.0
        return PortfolioTarget(
            weights={s: float(w.get(s, 0.0)) for s in universe},
            cash_weight=float(cash),
            turnover=float(turnover),
            expected_cost_bps=float(exp_cost),
            debug={"scale": scale, "vol_est": vol_est, "regime": regime},
        )

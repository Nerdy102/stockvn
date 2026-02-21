from __future__ import annotations

import datetime as dt
import json
from typing import Any

from core.raocmoe import (
    ExpertSet,
    GovernanceEngine,
    HedgeRouter,
    PortfolioController,
    RegimeEngine,
    UncertaintyEngine,
    build_default_detectors,
    load_config,
    update_all_detectors,
)


class RAOCMOEOverlay:
    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client
        self.cfg = load_config()
        self.detectors = build_default_detectors(self.cfg)
        self.regime = RegimeEngine(self.cfg)
        self.uq = UncertaintyEngine(self.cfg)
        self.experts = ExpertSet()
        self.router = HedgeRouter(
            regimes=list(self.cfg["regime"]["states"]),
            experts=list(self.cfg["moe"]["experts"]),
            eta=float(self.cfg["moe"]["routing"]["eta"]),
            loss_clip=float(self.cfg["moe"]["routing"]["loss_clip"]),
        )
        self.portfolio = PortfolioController(self.cfg)
        self.governance = GovernanceEngine(self.cfg, redis_client)
        self.t = 0

    def _redis_set(self, key: str, payload: str) -> None:
        if hasattr(self.redis, "set"):
            self.redis.set(key, payload)
            return
        if not hasattr(self.redis, "_kv"):
            self.redis._kv = {}
        self.redis._kv[key] = payload

    def on_bar_close(self, symbol: str, tf: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
        self.t += 1
        last = bars[-1]
        close = float(last.get("c", last.get("close", 0.0)))
        prev_close = (
            float(bars[-2].get("c", bars[-2].get("close", close))) if len(bars) > 1 else close
        )
        ret = (close - prev_close) / max(prev_close, 1e-8)
        realized_vol = abs(ret)
        corr_index = 0.2
        liq_stress = 0.1

        cp_events, cp_vector = update_all_detectors(
            self.detectors,
            {
                "mean_shift": ret,
                "vol_shift": realized_vol,
                "corr_shift": corr_index,
                "liq_shift": liq_stress,
            },
            self.t,
        )
        post = self.regime.update(
            t=self.t,
            features={
                "trend_strength": ret * 10.0,
                "realized_vol": realized_vol,
                "corr_index": corr_index,
                "liq_stress": liq_stress,
                "cpd_score": max(cp_vector.values()) if cp_vector else 0.0,
            },
            cp_events=cp_events,
        )

        ema20 = sum(float(b.get("c", b.get("close", close))) for b in bars[-20:]) / max(
            1, min(20, len(bars))
        )
        ema50 = sum(float(b.get("c", b.get("close", close))) for b in bars[-50:]) / max(
            1, min(50, len(bars))
        )
        vwap = sum(
            float(b.get("vwap", b.get("c", b.get("close", close)))) for b in bars[-20:]
        ) / max(1, min(20, len(bars)))

        factor_mu, _ = self.experts.factor_ml({"ret": ret, "realized_vol": realized_vol})
        trend_mu, _ = self.experts.trend_tech(close, ema20, ema50)
        mr_mu, _ = self.experts.meanrev_vwap(close, vwap, liq_stress)
        def_mu, _ = self.experts.defensive_lowvol(post.hard_regime)
        expert_mus = {
            "FACTOR_ML": factor_mu,
            "TREND_TECH": trend_mu,
            "MEANREV_VWAP": mr_mu,
            "DEFENSIVE_LOWVOL": def_mu,
        }
        moe = self.router.combine(post.hard_regime, expert_mus)

        intervals, uncertainty = self.uq.get_intervals(
            t=self.t,
            regime=post.hard_regime,
            symbol=symbol,
            sector="OTHER",
            yhat=moe.mu_hat_combined,
            realized_vol=max(1e-6, realized_vol),
        )
        self.uq.update_with_label(
            t=self.t,
            horizon="h1",
            y_true=ret,
            psi=0.0,
            cp_score=max(cp_vector.values()) if cp_vector else 0.0,
            sector="OTHER",
        )
        undercoverage, uc_details = self.uq.governance_undercoverage()

        gov = self.governance.evaluate(
            uq_undercoverage=undercoverage,
            psi=0.0,
            tca_shift=False,
            data_stale=False,
            extra={"coverage_breaches": uc_details},
        )

        target = self.portfolio.build_target(
            universe=[symbol],
            mu={symbol: moe.mu_hat_combined},
            uncertainty={symbol: uncertainty},
            prev={symbol: 0.0},
            vols={symbol: max(realized_vol, 0.001)},
            adv20={symbol: 1e12},
            nav=1.0,
            regime=post.hard_regime,
            paused=gov.paused,
        )
        h1 = next(iv for iv in intervals if iv.horizon == "h1")
        snap = {
            "symbol": symbol,
            "tf": tf,
            "end_ts": str(last["end_ts"]),
            "regime_posterior": post.posterior,
            "hard_regime": post.hard_regime,
            "mu_hat": moe.mu_hat_combined,
            "uncertainty": uncertainty,
            "interval_h1": {"lower": h1.lower, "upper": h1.upper, "alpha": h1.alpha},
            "routing_weights": moe.weights,
            "expert_mus": moe.expert_mus,
            "governance_status": gov.to_json(),
            "recommended_target_weight": float(target.weights.get(symbol, 0.0)),
            "debug_fields": {
                "cp_events_count": len(cp_events),
                "cp_score_vector": cp_vector,
                "routing_entropy": moe.entropy,
            },
            "paper_only": True,
        }
        encoded = json.dumps(snap, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._redis_set(f"realtime:signals:{symbol}:{tf}", encoded)
        self._redis_set(
            "realtime:ops:summary",
            json.dumps(
                {
                    "last_update": dt.datetime.utcnow().isoformat() + "Z",
                    "signals_count": 1,
                    "cp_events_count": len(cp_events),
                    "pause_flag": gov.paused,
                    "lag": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self._redis_set(
            f"realtime:state:raocmoe:{symbol}:{tf}",
            json.dumps(
                {
                    "t": self.t,
                    "hard_regime": post.hard_regime,
                    "last_mu": moe.mu_hat_combined,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return snap

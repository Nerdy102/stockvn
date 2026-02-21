from __future__ import annotations

import math

from core import market_rules as mr
from core.cost_model import FillConfig, SlippageConfig, calc_fill_ratio, calc_slippage_bps

from .cpd import MeanShiftDetector
from .types import ExecutionEstimate


class ExecutionTCA:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["execution"]
        self.lr = float(self.cfg["tca"]["lr"])
        self.delta = float(self.cfg["tca"]["huber_delta"])
        self.beta: dict[str, list[float]] = {}
        self.cpd = MeanShiftDetector(
            name="tca_residual",
            robust_clip=8.0,
            threshold_base=float(self.cfg["tca_cpd"]["threshold_base"]),
            cooldown_bars=int(self.cfg["tca_cpd"]["cooldown_bars"]),
            geometric_base=2,
            max_candidates=64,
        )

    def _feature_vector(
        self,
        participation: float,
        atr_pct: float,
        spread_proxy: float,
        session_flag: float,
        regime_id: float,
        limit_distance_ticks: float,
    ) -> list[float]:
        return [
            1.0,
            participation,
            atr_pct,
            spread_proxy,
            session_flag,
            regime_id,
            limit_distance_ticks,
        ]

    def _predict_slippage(self, symbol: str, x: list[float]) -> float:
        b = self.beta.get(symbol)
        if b is None:
            b = [0.0 for _ in x]
            self.beta[symbol] = b
        return float(sum(bi * xi for bi, xi in zip(b, x)))

    def _update_huber(self, symbol: str, x: list[float], y: float) -> None:
        pred = self._predict_slippage(symbol, x)
        resid = pred - y
        grad_scale = (
            resid if abs(resid) <= self.delta else self.delta * (1.0 if resid > 0 else -1.0)
        )
        b = self.beta[symbol]
        for i in range(len(b)):
            b[i] -= self.lr * grad_scale * x[i]

    def estimate(
        self,
        symbol: str,
        side: str,
        delta_weight: float,
        nav: float,
        price: float,
        adtv: float,
        atr14: float,
        *,
        instrument_type: str,
        reference_price: float | None = None,
        at_upper_limit: bool = False,
        at_lower_limit: bool = False,
        spread_proxy: float = 0.0,
        session_flag: float = 1.0,
        regime_id: int = 0,
        limit_distance_ticks: float = 10.0,
    ) -> ExecutionEstimate:
        notional = abs(delta_weight) * nav
        qty = notional / max(1e-8, price)
        if instrument_type == "stock":
            qty = float(mr.clamp_qty_to_board_lot(int(qty), board_lot=100))
        rounded = mr.round_price(price, side=side, instrument_type=instrument_type)
        if instrument_type == "stock" and reference_price is not None:
            valid = mr.load_market_rules("configs/market_rules_vn.yaml").validate_price_limit(
                price=rounded,
                reference_price=float(reference_price),
                context="normal",
            )
            if not valid:
                lower, upper = mr.calc_price_limits(float(reference_price), "normal")
                rounded = min(max(rounded, lower), upper)
        slippage_bps = calc_slippage_bps(notional, adtv, atr14, price, SlippageConfig())
        fill = calc_fill_ratio(
            side,
            "MARKET",
            at_upper_limit=at_upper_limit,
            at_lower_limit=at_lower_limit,
            cfg=FillConfig(
                limit_up_buy_penalty=float(
                    self.cfg["limit_fill_penalty"]["buy_at_upper_limit_mult"]
                ),
                limit_down_sell_penalty=float(
                    self.cfg["limit_fill_penalty"]["sell_at_lower_limit_mult"]
                ),
            ),
        )
        participation = notional / max(1e-8, adtv)
        atr_pct = atr14 / max(1e-8, price)
        x = self._feature_vector(
            participation=float(participation),
            atr_pct=float(atr_pct),
            spread_proxy=float(spread_proxy),
            session_flag=float(session_flag),
            regime_id=float(regime_id),
            limit_distance_ticks=float(limit_distance_ticks),
        )
        pred_bps = self._predict_slippage(symbol, x)
        return ExecutionEstimate(
            symbol=symbol,
            qty=float(qty),
            exec_price=float(rounded),
            fill_ratio=float(fill),
            slippage_bps=float(slippage_bps),
            predicted_slippage_bps=float(pred_bps),
        )

    def update_realized(
        self,
        t: int,
        symbol: str,
        features: list[float],
        realized_slippage_bps: float,
        predicted_slippage_bps: float,
    ) -> bool:
        self._update_huber(symbol=symbol, x=features, y=realized_slippage_bps)
        cp, _, _ = self.cpd.update(realized_slippage_bps - predicted_slippage_bps, t)
        return bool(cp)

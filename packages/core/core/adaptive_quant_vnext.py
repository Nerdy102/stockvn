from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from contracts.canonical import derive_event_id, hash_payload


def _safe_std(v: pd.Series | np.ndarray, floor: float = 1e-6) -> float:
    x = np.asarray(v, dtype=float)
    s = float(np.std(x)) if x.size else 0.0
    return max(s, floor)


def _z_shift(pre: np.ndarray, post: np.ndarray) -> float:
    if pre.size == 0 or post.size == 0:
        return 0.0
    num = abs(float(np.mean(post) - np.mean(pre)))
    den = max(float(np.std(np.concatenate([pre, post]))), 1e-9)
    return num / den


def _var_shift(pre: np.ndarray, post: np.ndarray) -> float:
    v1 = max(float(np.var(pre)), 1e-12)
    v2 = max(float(np.var(post)), 1e-12)
    return abs(math.log(v2 / v1))


def _alpha_threshold(alpha_k: float, base: float) -> float:
    # Tail-like bound approximation without scipy dependency.
    return base + math.sqrt(max(0.0, 2.0 * math.log(1.0 / max(alpha_k, 1e-12))))


@dataclass(frozen=True)
class CPEvent:
    series_key: str
    tf: str
    detected_at_index: int
    cp_type: str
    stat: float
    threshold: float
    severity: str
    candidates_checked: int
    window_short: int
    window_long: int
    metadata_json: dict[str, Any]


@dataclass(frozen=True)
class CPMonitorInput:
    returns: Iterable[float]
    corr_summary: Iterable[float] | None = None
    spread_proxy: Iterable[float] | None = None
    volume_z: Iterable[float] | None = None
    limit_rate: Iterable[float] | None = None


class OnlineGridCPD:
    def __init__(self, max_window: int = 128, alpha_total: float = 0.01) -> None:
        self.max_window = int(max_window)
        self.alpha_total = float(alpha_total)

    def candidate_windows(self, n: int) -> list[int]:
        out: list[int] = []
        w = 1
        while w <= min(self.max_window, max(1, n // 2)):
            out.append(w)
            w *= 2
        return out

    def _liquidity_shift(self, w: int, spread: np.ndarray, volz: np.ndarray, lim: np.ndarray) -> float:
        if min(spread.size, volz.size, lim.size) < 2 * w:
            return 0.0
        pre = np.column_stack((spread[-2 * w : -w], volz[-2 * w : -w], lim[-2 * w : -w]))
        post = np.column_stack((spread[-w:], volz[-w:], lim[-w:]))
        mu_pre = np.mean(pre, axis=0)
        mu_post = np.mean(post, axis=0)
        sigma = np.std(np.vstack([pre, post]), axis=0)
        sigma = np.where(sigma < 1e-9, 1e-9, sigma)
        return float(np.linalg.norm((mu_post - mu_pre) / sigma, ord=2))

    def detect(self, monitor: CPMonitorInput, tf: str = "60m", series_key: str = "VNINDEX") -> CPEvent | None:
        returns = np.asarray(list(monitor.returns), dtype=float)
        n = returns.size
        if n < 12:
            return None

        def _arr(x: Iterable[float] | None) -> np.ndarray:
            return np.asarray(list(x), dtype=float) if x is not None else np.asarray([], dtype=float)

        corr = _arr(monitor.corr_summary)
        spread = _arr(monitor.spread_proxy)
        volz = _arr(monitor.volume_z)
        lim = _arr(monitor.limit_rate)

        candidates = self.candidate_windows(n)
        best: CPEvent | None = None
        for idx, w in enumerate(candidates, start=1):
            if n < 4 * w or w < 4:
                continue
            pre_r = returns[-2 * w : -w]
            post_r = returns[-w:]
            alpha_k = self.alpha_total / (idx * (idx + 1))

            tests: list[tuple[str, float, float]] = []

            mean_stat = _z_shift(pre_r, post_r)
            tests.append(("MEAN", mean_stat, _alpha_threshold(alpha_k, base=1.8)))

            var_stat = _var_shift(pre_r**2, post_r**2)
            tests.append(("VAR", var_stat, _alpha_threshold(alpha_k, base=1.6)))

            if corr.size >= 4 * w:
                pre_c = corr[-2 * w : -w]
                post_c = corr[-w:]
                corr_stat = _z_shift(pre_c, post_c)
                tests.append(("CORR", corr_stat, _alpha_threshold(alpha_k, base=1.8)))

            if spread.size and volz.size and lim.size:
                liq_stat = self._liquidity_shift(w, spread, volz, lim)
                tests.append(("LIQ", liq_stat, _alpha_threshold(alpha_k, base=1.8)))

            for cp_type, stat, threshold in tests:
                if stat >= threshold:
                    event = CPEvent(
                        series_key=series_key,
                        tf=tf,
                        detected_at_index=n - 1,
                        cp_type=cp_type,
                        stat=float(stat),
                        threshold=float(threshold),
                        severity="HIGH",
                        candidates_checked=len(candidates),
                        window_short=w,
                        window_long=2 * w,
                        metadata_json={
                            "candidate_index": idx,
                            "alpha_k": alpha_k,
                            "cp_offset": w,
                        },
                    )
                    if best is None or event.stat > best.stat:
                        best = event
        return best

    def detect_variance_shift(self, returns: Iterable[float], tf: str = "60m", series_key: str = "VNINDEX") -> CPEvent | None:
        return self.detect(CPMonitorInput(returns=returns), tf=tf, series_key=series_key)


def build_cp_event_payload(cp_row_id: int, severity: str, cp_type: str, series_key: str) -> dict[str, Any]:
    payload = {
        "event_type": "CP_EVENT",
        "cp_id": int(cp_row_id),
        "severity": str(severity),
        "cp_type": str(cp_type),
        "series_key": str(series_key),
    }
    payload["event_id"] = derive_event_id(payload)
    payload["payload_hash"] = hash_payload(payload)
    return payload


REGIME_R0 = "TREND_UP"
REGIME_R1 = "SIDEWAYS"
REGIME_R2 = "RISK_OFF"
REGIME_R3 = "PANIC_VOL"


@dataclass(frozen=True)
class RegimeState:
    probs: dict[str, float]
    active_regime: str
    hysteresis_applied: bool


class RegimeEngineV2:
    def __init__(self) -> None:
        self.prev_regime = REGIME_R1

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        z = scores - np.max(scores)
        e = np.exp(z)
        return e / np.sum(e)

    def infer(self, features: dict[str, float], cp_recent: int) -> RegimeState:
        f1 = float(features.get("f1", 0.0))
        f2 = float(features.get("f2", 0.0))
        f3 = float(features.get("f3", 0.0))
        f4 = float(features.get("f4", 0.0))
        f5 = float(features.get("f5", 0.0))
        f6 = float(cp_recent)

        s0 = +1.5 * f1 + 1.0 * f2 - 0.5 * f3 + 0.3 * f5 - 0.8 * f6
        s1 = -0.2 * abs(f1) + 0.2 * (1 - f2) - 0.2 * f3 + 0.2 * abs(f5)
        s2 = -1.2 * f2 + 0.8 * f3 - 0.6 * f5 + 0.9 * f6 + 0.6 * (1.0 if f4 < -0.03 else 0.0)
        s3 = +1.4 * f3 + 1.2 * (1.0 if f4 < -0.05 else 0.0) + 1.0 * f6 - 0.8 * f5
        probs = self.softmax(np.array([s0, s1, s2, s3], dtype=float))
        keys = [REGIME_R0, REGIME_R1, REGIME_R2, REGIME_R3]
        p = {k: float(v) for k, v in zip(keys, probs)}
        arg = keys[int(np.argmax(probs))]
        hysteresis_applied = False
        active = arg
        if p[REGIME_R3] >= 0.60:
            active = REGIME_R3
        elif p[arg] < 0.55:
            active = self.prev_regime
            hysteresis_applied = True
        self.prev_regime = active
        return RegimeState(probs=p, active_regime=active, hysteresis_applied=hysteresis_applied)




@dataclass(frozen=True)
class RegimeFeatures:
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: int


def compute_regime_features_v2(
    close: pd.Series,
    breadth: pd.Series,
    cp_events_recent: pd.Series | None = None,
    annualization: float = 252.0 * 6.5,
) -> pd.DataFrame:
    """Compute locked L2 regime features on VNINDEX 60m/daily series."""
    c = close.astype(float).copy()
    b = breadth.astype(float).reindex(c.index).fillna(0.0)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    spread = ema20 - ema50
    f1 = ((spread - spread.rolling(60, min_periods=10).mean()) / spread.rolling(60, min_periods=10).std().replace(0.0, np.nan)).fillna(0.0)
    f2 = (c > ema50).astype(float)
    ret = c.pct_change().fillna(0.0)
    rv = ret.rolling(26, min_periods=8).std() * math.sqrt(float(annualization))
    f3 = ((rv - rv.rolling(60, min_periods=10).mean()) / rv.rolling(60, min_periods=10).std().replace(0.0, np.nan)).fillna(0.0)
    rolling_max = c.rolling(26, min_periods=1).max()
    f4 = (c / rolling_max - 1.0).fillna(0.0)
    f5 = b.clip(-1.0, 1.0)
    if cp_events_recent is None:
        f6 = pd.Series(0, index=c.index, dtype=int)
    else:
        cp = cp_events_recent.reindex(c.index).fillna(0).astype(int)
        f6 = cp.rolling(26, min_periods=1).max().astype(int)
    out = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "f6": f6,
    }, index=c.index)
    return out


@dataclass(frozen=True)
class RegimePolicyAction:
    governance_risk_level: str
    gross_exposure_cap_multiplier: float
    min_cash_buffer: float
    restore_progress: float


class RegimePolicyEngine:
    def __init__(self) -> None:
        self._r0_streak = 0
        self._restore_step = 0

    def action_for(self, active_regime: str) -> RegimePolicyAction:
        if active_regime == REGIME_R2:
            self._r0_streak = 0
            self._restore_step = 0
            return RegimePolicyAction("HIGH", 0.5, 0.20, 0.0)
        if active_regime == REGIME_R3:
            self._r0_streak = 0
            self._restore_step = 0
            return RegimePolicyAction("HIGH", 0.25, 0.35, 0.0)

        if active_regime == REGIME_R0:
            self._r0_streak += 1
            if self._r0_streak >= 26:
                self._restore_step = min(26, self._restore_step + 1)
        else:
            self._r0_streak = 0
            self._restore_step = 0

        progress = self._restore_step / 26.0
        cap = 0.5 + 0.5 * progress
        cash = 0.20 - 0.15 * progress
        return RegimePolicyAction("NORMAL", float(cap), float(cash), float(progress))


class ACIUncertainty:
    def __init__(self, alpha: float = 0.20, eta: float = 0.05, window: int = 200) -> None:
        self.alpha = alpha
        self.eta = eta
        self.window = window
        self.residuals: deque[float] = deque(maxlen=window)
        self.q_t = 0.01

    def bootstrap(self, residuals: Iterable[float]) -> None:
        vals = np.abs(np.asarray(list(residuals), dtype=float))
        for v in vals[-self.window :]:
            self.residuals.append(float(v))
        if vals.size >= self.window:
            self.q_t = float(np.quantile(vals[-self.window :], 0.80))

    def update(self, residual: float) -> tuple[float, float]:
        hit = 1 if abs(float(residual)) <= self.q_t else 0
        self.alpha = float(np.clip(self.alpha + self.eta * (0.20 - (1 - hit)), 0.05, 0.50))
        self.residuals.append(abs(float(residual)))
        if self.residuals:
            self.q_t = float(np.quantile(np.asarray(self.residuals), 1 - self.alpha))
        return self.alpha, self.q_t


class CPTCSwitchingCalibrator:
    def __init__(self) -> None:
        self.s0: deque[float] = deque(maxlen=200)
        self.s1: deque[float] = deque(maxlen=60)
        self.state = "S0"
        self.cooldown_until_index = -1

    def update(self, residual: float, idx: int, cp_recent: bool, risk_regime: bool) -> tuple[str, float]:
        a = abs(float(residual))
        self.s0.append(a)
        self.s1.append(a)
        if (cp_recent or risk_regime) and self.state != "S1":
            self.state = "S1"
            self.cooldown_until_index = idx + 26
        elif self.state == "S1" and idx >= self.cooldown_until_index and not (cp_recent or risk_regime):
            self.state = "S0"
        active = self.s1 if self.state == "S1" else self.s0
        q = float(np.quantile(np.asarray(active), 0.80)) if active else 0.01
        return self.state, q


def sector_pooled_q(q_symbol_local: float, q_sector: float) -> float:
    return float(max(float(q_symbol_local), float(q_sector)))


def base_return_forecast(close: pd.Series) -> pd.Series:
    c = close.astype(float)
    momentum_1 = c.pct_change(3).fillna(0.0)
    ema20 = c.ewm(span=20, adjust=False).mean()
    mean_reversion_1 = (-(c - ema20) / ema20.replace(0.0, np.nan)).fillna(0.0)
    return 0.6 * momentum_1 + 0.4 * mean_reversion_1


def residual_series(realized_return: pd.Series, r_hat: pd.Series) -> pd.Series:
    rr = realized_return.astype(float)
    rh = r_hat.astype(float).reindex(rr.index).fillna(0.0)
    return rr - rh


class SectorResidualPool:
    def __init__(self, window: int = 200) -> None:
        self.window = int(window)
        self._windows: dict[str, deque[float]] = {}

    def update(self, sector: str, residual: float) -> float:
        key = str(sector)
        if key not in self._windows:
            self._windows[key] = deque(maxlen=self.window)
        self._windows[key].append(abs(float(residual)))
        vals = np.asarray(self._windows[key], dtype=float)
        return float(np.quantile(vals, 0.80)) if vals.size else 0.01


def uncertainty_metrics(
    y_true: pd.Series,
    lo: pd.Series,
    hi: pd.Series,
    p_outperform: pd.Series | None = None,
    window: int = 500,
) -> dict[str, float]:
    y = y_true.astype(float).tail(window)
    l = lo.astype(float).reindex(y.index)
    h = hi.astype(float).reindex(y.index)
    inside = ((y >= l) & (y <= h)).astype(float)
    widths = (h - l).clip(lower=0.0)
    out: dict[str, float] = {
        "coverage": float(inside.mean()) if len(inside) else 0.0,
        "width_median": float(widths.median()) if len(widths) else 0.0,
        "width_p90": float(widths.quantile(0.90)) if len(widths) else 0.0,
        "ece": 0.0,
    }
    if p_outperform is not None and len(y) > 0:
        p = p_outperform.astype(float).reindex(y.index).clip(0.0, 1.0)
        y_bin = (y > 0.0).astype(float)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= bins[i + 1])
            if m.any():
                conf = float(p[m].mean())
                acc = float(y_bin[m].mean())
                w = float(m.mean())
                ece += abs(acc - conf) * w
        out["ece"] = float(ece)
    return out


def governance_from_coverage(coverage_series: pd.Series, lookback_days: int = 20) -> str:
    c = coverage_series.astype(float).tail(lookback_days)
    if len(c) < lookback_days:
        return "OK"
    if bool((c < 0.70).all()):
        return "PAUSE"
    if bool((c < 0.75).all()):
        return "WARNING"
    return "OK"


def pit_daily_matured_only(df: pd.DataFrame) -> pd.DataFrame:
    if "matured_date" not in df.columns or "as_of_date" not in df.columns:
        return df.copy()
    out = df.copy()
    return out.loc[out["matured_date"] <= out["as_of_date"]].copy()


def zscore_cross_section(scores: pd.Series) -> pd.Series:
    s = scores.astype(float)
    return (s - s.mean()) / _safe_std(s)


def update_expert_weights(
    weights: np.ndarray,
    expert_scores: pd.DataFrame,
    realized_next_return: pd.Series,
    eta: float = 0.20,
    allow_zero_breakout: bool = False,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    rewards = []
    r = realized_next_return.rank(method="average")
    r_std = float(r.std())
    for c in expert_scores.columns:
        s = expert_scores[c].rank(method="average")
        s_std = float(s.std())
        if s_std <= 1e-12 or r_std <= 1e-12:
            reward = 0.0
        else:
            corr = float(((s - s.mean()) * (r - r.mean())).mean() / (s_std * r_std))
            reward = float(np.clip(corr, -1.0, 1.0))
        rewards.append(reward)
    rewards_arr = np.asarray(rewards, dtype=float)
    w = w * np.exp(eta * rewards_arr)
    w = w / np.sum(w)
    mins = np.array([0.02, 0.02, 0.00 if allow_zero_breakout else 0.02, 0.02, 0.02])
    maxs = np.array([0.70, 0.70, 0.70, 0.70, 0.70])
    w = np.minimum(np.maximum(w, mins), maxs)
    w = w / np.sum(w)
    return w




EXPERTS = ["E0_FACTOR", "E1_TREND", "E2_BREAKOUT", "E3_MEAN_REVERSION", "E4_DEFENSIVE"]
REGIME_INIT_WEIGHTS: dict[str, list[float]] = {
    REGIME_R0: [0.35, 0.25, 0.20, 0.10, 0.10],
    REGIME_R1: [0.35, 0.15, 0.10, 0.25, 0.15],
    REGIME_R2: [0.40, 0.10, 0.05, 0.15, 0.30],
    REGIME_R3: [0.45, 0.05, 0.00, 0.10, 0.40],
}


def init_regime_weights() -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) for k, v in REGIME_INIT_WEIGHTS.items()}


def build_expert_scores_v2(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    value = d.get("value", pd.Series(0.0, index=d.index)).astype(float)
    quality = d.get("quality", pd.Series(0.0, index=d.index)).astype(float)
    momentum = d.get("momentum", pd.Series(0.0, index=d.index)).astype(float)
    low_vol = d.get("low_vol", pd.Series(0.0, index=d.index)).astype(float)
    dividend = d.get("dividend", pd.Series(0.0, index=d.index)).astype(float)
    beta = d.get("beta", pd.Series(0.0, index=d.index)).astype(float)
    trend_setup = d.get("trend_setup", pd.Series(0, index=d.index)).astype(float)
    breakout_setup = d.get("breakout_setup", pd.Series(0, index=d.index)).astype(float)
    ema50_slope = d.get("ema50_slope", pd.Series(0.0, index=d.index)).astype(float).clip(-1.0, 1.0)
    return_3bars = d.get("return_3bars", pd.Series(0.0, index=d.index)).astype(float)
    far_above_ema20 = d.get("far_above_ema20", pd.Series(0, index=d.index)).astype(float)
    far_below_ema20 = d.get("far_below_ema20", pd.Series(0, index=d.index)).astype(float)

    score_factor = (value + quality + momentum + low_vol + dividend) / 5.0
    score_trend = trend_setup + ema50_slope
    score_breakout = breakout_setup
    rz = zscore_cross_section(return_3bars)
    score_mr = (-rz * far_above_ema20 + rz * far_below_ema20).clip(-3.0, 3.0)
    score_def = zscore_cross_section(low_vol) + zscore_cross_section(dividend) - zscore_cross_section(beta)

    raw = pd.DataFrame(
        {
            "E0_FACTOR": score_factor,
            "E1_TREND": score_trend,
            "E2_BREAKOUT": score_breakout,
            "E3_MEAN_REVERSION": score_mr,
            "E4_DEFENSIVE": score_def,
        },
        index=d.index,
    )
    return raw.apply(zscore_cross_section)


def combine_alpha_score(expert_scores_z: pd.DataFrame, regime: str, regime_weights: dict[str, np.ndarray]) -> pd.Series:
    w = np.asarray(regime_weights[regime], dtype=float)
    cols = [c for c in EXPERTS if c in expert_scores_z.columns]
    w = w[: len(cols)]
    return expert_scores_z[cols].mul(w, axis=1).sum(axis=1)


def update_regime_weights_for_next_bar(
    regime_weights: dict[str, np.ndarray],
    regime: str,
    scores_at_t: pd.DataFrame,
    realized_next_return: pd.Series,
    eta: float = 0.20,
    tf: str = "60m",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if tf != "60m":
        return regime_weights, {"updated": False, "reason": "only_60m"}
    w_old = np.asarray(regime_weights[regime], dtype=float)
    w_new = update_expert_weights(
        w_old,
        scores_at_t[EXPERTS],
        realized_next_return,
        eta=eta,
        allow_zero_breakout=(regime == REGIME_R3),
    )
    out = {k: np.asarray(v, dtype=float).copy() for k, v in regime_weights.items()}
    out[regime] = w_new
    return out, {"updated": True, "regime": regime, "weights_old": w_old.tolist(), "weights_new": w_new.tolist()}


def alpha_prediction_payload(symbol: str, tf: str, end_ts: datetime, alpha_score: float, alpha_confidence: float, regime: str, weights_used: list[float]) -> dict[str, Any]:
    payload = {
        "symbol": str(symbol),
        "tf": str(tf),
        "end_ts": end_ts.isoformat(),
        "alpha_score": float(alpha_score),
        "alpha_confidence": float(alpha_confidence),
        "regime": str(regime),
        "weights_used_json": [float(x) for x in weights_used],
    }
    payload["prediction_hash"] = hash_payload(payload)
    return payload

def alpha_shrink(alpha_score: float, q_symbol: float, q_threshold_high: float) -> tuple[float, float]:
    scaled = float(alpha_score) * (0.5 if float(q_symbol) > float(q_threshold_high) else 1.0)
    conf = 1.0 / (1.0 + float(q_symbol))
    return scaled, conf


def benefit_cost_gate(alpha_score_z: float, forecast_std: float, position_value: float, cost_rate: float, turnover_notional: float) -> bool:
    benefit = float(alpha_score_z) * float(forecast_std) * float(position_value)
    cost = float(cost_rate) * float(turnover_notional)
    return benefit >= 2.0 * cost


RUNBOOK_UNDERCOVERAGE = "RB-UNC-001"
RUNBOOK_SLO = "RB-SLO-005"
RUNBOOK_RECON = "RB-REC-002"
RUNBOOK_DD = "RB-DD-003"


def log_tca_fill(
    decision_ts: datetime,
    submit_ts: datetime,
    fill_ts: datetime,
    intended_price: float,
    executed_price: float,
    notional: float,
    participation_rate: float,
    session: str,
    regime: str,
) -> dict[str, Any]:
    side_sign = 1.0 if executed_price >= intended_price else -1.0
    slippage_bps = side_sign * (abs(float(executed_price) - float(intended_price)) / max(float(intended_price), 1e-9)) * 1e4
    return {
        "decision_ts": decision_ts.isoformat(),
        "submit_ts": submit_ts.isoformat(),
        "fill_ts": fill_ts.isoformat(),
        "intended_price": float(intended_price),
        "executed_price": float(executed_price),
        "notional": float(notional),
        "participation_rate": float(participation_rate),
        "slippage_bps": float(slippage_bps),
        "session": str(session),
        "regime": str(regime),
    }


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residuals)
    w = np.ones_like(a)
    m = a > delta
    w[m] = delta / a[m]
    return w


def fit_tca_huber_regression(df: pd.DataFrame, delta: float = 1.5, max_iter: int = 20) -> dict[str, Any]:
    cols = ["x_notional_adtv", "x_atr_price", "x_limit_day", "x_session", "x_regime"]
    y = df["slippage_bps"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[c].astype(float).to_numpy() for c in cols])
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        r = y - X @ beta
        w = _huber_weights(r, delta)
        W = np.diag(w)
        beta_new = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
        if float(np.max(np.abs(beta_new - beta))) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    pred = X @ beta
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
    params = {
        "b0": float(beta[0]),
        "b1": float(beta[1]),
        "b2": float(beta[2]),
        "b3": float(beta[3]),
        "b4": float(beta[4]),
        "b5": float(beta[5]),
    }
    metrics = {"rmse": rmse, "n": int(len(df))}
    return {"params": params, "metrics": metrics}


def tca_params_record(bucket_key: str, as_of_date: str, params: dict[str, float], metrics: dict[str, float]) -> dict[str, Any]:
    payload = {
        "bucket_key": str(bucket_key),
        "date": str(as_of_date),
        "params_json": params,
        "metrics_json": metrics,
    }
    payload["param_hash"] = hash_payload(payload)
    return payload


def detect_impact_regime_shift_from_residuals(residuals: Iterable[float]) -> bool:
    cpd = OnlineGridCPD(max_window=128, alpha_total=0.01)
    ev = cpd.detect(CPMonitorInput(returns=list(residuals)), tf="1D", series_key="TCA_RESID")
    return ev is not None


def impact_shift_adjustment(shift_detected: bool, base_bps: float) -> dict[str, Any]:
    if not shift_detected:
        return {"base_bps": float(base_bps), "tighten_fill_probability": False, "tag": "STABLE"}
    return {"base_bps": float(base_bps + 5.0), "tighten_fill_probability": True, "tag": "IMPACT_REGIME_SHIFT"}


def realtime_actions_for_bar(event_type: str, timeframe: str) -> list[str]:
    et = str(event_type).upper()
    tf = str(timeframe).lower()
    if et != "BAR_CLOSE":
        return []
    if tf == "15m":
        return ["update_incremental_indicators", "update_setups"]
    if tf == "60m":
        return [
            "update_incremental_indicators",
            "compute_regime",
            "update_expert_weights",
            "generate_alpha",
            "run_ocpd",
            "update_uncertainty",
            "update_hot_cache",
        ]
    return []


def daily_eod_actions() -> list[str]:
    return [
        "recompute_daily_factors",
        "train_listnet_if_applicable",
        "update_uncertainty_metrics",
        "calibrate_tca",
        "governance_evaluation",
        "generate_daily_report_pack",
    ]


def evaluate_governance_triggers(
    coverage_series: pd.Series,
    signal_latency_p95: float,
    reconciliation_mismatch: bool,
    intraday_drawdown_breach: bool,
) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    status = governance_from_coverage(coverage_series)
    if status == "PAUSE":
        incidents.append({"level": "PAUSE", "reason": "undercoverage", "runbook_id": RUNBOOK_UNDERCOVERAGE})
    elif status == "WARNING":
        incidents.append({"level": "WARNING", "reason": "undercoverage", "runbook_id": RUNBOOK_UNDERCOVERAGE})
    if float(signal_latency_p95) > 5.0:
        incidents.append({"level": "WARNING", "reason": "slo_latency", "runbook_id": RUNBOOK_SLO})
    if bool(reconciliation_mismatch):
        incidents.append({"level": "PAUSE", "reason": "reconciliation_mismatch", "runbook_id": RUNBOOK_RECON})
    if bool(intraday_drawdown_breach):
        incidents.append({"level": "PAUSE", "reason": "intraday_drawdown", "runbook_id": RUNBOOK_DD})
    return incidents


def bounded_series(values: pd.Series, max_points: int = 200) -> pd.Series:
    if len(values) <= max_points:
        return values
    idx = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return values.iloc[idx]


def reaction_time_to_cp(cp_index: int, regime_change_index: int) -> int:
    return max(0, int(regime_change_index) - int(cp_index))

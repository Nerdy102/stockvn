from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


@dataclass
class CPScore:
    detector: str
    score: float


@dataclass
class CPEvent:
    detector_name: str
    tau_hat: int | None
    score: float
    timestamp: int
    cp_scores: list[CPScore] = field(default_factory=list)


@dataclass
class RegimePosterior:
    posterior: dict[str, float]
    hard_regime: str
    time_since_last_switch: int
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class UQInterval:
    horizon: str
    lower: float
    upper: float
    alpha: float


@dataclass
class ExpertOutput:
    expert: str
    mu_hat: float
    confidence: float


@dataclass
class MoEOutput:
    mu_hat_combined: float
    expert_mus: dict[str, float]
    weights: dict[str, float]
    entropy: float


@dataclass
class PortfolioTarget:
    weights: dict[str, float]
    cash_weight: float
    turnover: float
    expected_cost_bps: float
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionEstimate:
    symbol: str
    qty: float
    exec_price: float
    fill_ratio: float
    slippage_bps: float
    predicted_slippage_bps: float


@dataclass
class GovernanceStatus:
    paused: bool
    pause_reason: str | None
    last_change_ts: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return _sanitize(asdict(self))

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestReport:
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    expectancy: float


DISCLAIMER = (
    "Hiệu quả quá khứ không đảm bảo tương lai; có thể overfit; "
    "phụ thuộc chi phí/độ khớp; có rủi ro."
)


def _safe_std(x: pd.Series) -> float:
    s = float(x.std()) if len(x) > 1 else 0.0
    return s if math.isfinite(s) else 0.0


def build_report(equity_curve: pd.Series, returns: pd.Series) -> dict[str, Any]:
    r = returns.dropna()
    if r.empty:
        rep = BacktestReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return {**rep.__dict__, "disclaimer": DISCLAIMER}

    periods = max(1, len(r))
    years = periods / 252.0
    cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(years, 1e-9)) - 1)

    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1
    mdd = float(dd.min())

    mu = float(r.mean())
    sd = _safe_std(r)
    neg = r[r < 0]
    sdn = _safe_std(neg)

    sharpe = float((mu / sd) * math.sqrt(252)) if sd > 0 else 0.0
    sortino = float((mu / sdn) * math.sqrt(252)) if sdn > 0 else 0.0

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(abs(r[r < 0].sum()))
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

    win_rate = float((r > 0).mean())
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(abs(r[r < 0].mean())) if (r < 0).any() else 0.0
    expectancy = float(win_rate * avg_win - (1 - win_rate) * avg_loss)

    rep = BacktestReport(cagr, mdd, sharpe, sortino, pf, expectancy)
    return {**rep.__dict__, "disclaimer": DISCLAIMER}

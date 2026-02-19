from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from core.simple_mode.models import run_signal
from core.simple_mode.schemas import BacktestReport


def _hash_obj(v: object) -> str:
    s = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def quick_backtest(
    model_id: str,
    symbol: str,
    df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    *,
    position_mode: str = "long_only",
) -> BacktestReport:
    if "date" in df.columns:
        w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    else:
        w = df.copy()
    strat = pd.Series(dtype=float)
    if len(w) < 30:
        net = 0.0
        vol = 0.0
        mdd = 0.0
        turnover = 0.0
    else:
        sig = run_signal(model_id, symbol, "1D", w)
        rets = w["close"].pct_change().fillna(0.0)
        if sig.proposed_side == "BUY":
            pos = 1.0
        elif sig.proposed_side in {"SELL", "SHORT"} and position_mode == "long_short":
            pos = -1.0
        else:
            pos = 0.0
        strat = rets * pos
        equity = (1 + strat).cumprod()
        peak = equity.cummax()
        dd = equity / peak - 1
        mdd = float(dd.min()) if not dd.empty else 0.0
        net = float(equity.iloc[-1] - 1)
        vol = float(strat.std() * np.sqrt(252)) if float(strat.std()) > 0 else 0.0
        turnover = float(abs(pos))

    sharpe = float((net / vol) if vol > 0 else 0.0)
    sortino = sharpe
    cagr = float((1 + net) ** (252 / max(len(w), 1)) - 1)

    long_exposure = float((strat > 0).mean()) if len(w) else 0.0
    short_exposure = float((strat < 0).mean()) if len(w) else 0.0

    conf = {
        "model_id": model_id,
        "symbol": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "position_mode": position_mode,
    }
    dataset_hash = _hash_obj({"rows": len(w), "head": w.head(3).to_dict(orient="records")})
    return BacktestReport(
        model_id=model_id,
        symbols=[symbol],
        start=start,
        end=end,
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        sortino=sortino,
        turnover=turnover,
        net_return=net,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        config_hash=_hash_obj(conf),
        dataset_hash=dataset_hash,
        code_hash="simple_mode_v1",
    )

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from research.strategies.user_v1_stability_pack import _smoothed_scores


def _load_costs() -> tuple[float, float, float]:
    cfg = yaml.safe_load(Path("configs/eval_lab.yaml").read_text(encoding="utf-8"))
    ex = cfg["execution"]
    return float(ex["commission_bps"]), float(ex["sell_tax_bps"]), float(ex["slippage_bps"])


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    commission_bps, sell_tax_bps, slippage_bps = _load_costs()
    top_k = 3
    buffer = 2
    no_trade_band = 0.02
    f = _smoothed_scores(frame)
    f["atr_pct"] = (
        ((f["high"] - f["low"]).abs() / f["close"].replace(0.0, 1.0))
        .rolling(14)
        .mean()
        .fillna(0.001)
    )
    f["slippage_bps_est"] = (
        (0.5 * f["atr_pct"] * 10000.0).clip(lower=10.0).fillna(max(10.0, slippage_bps))
    )
    f["est_cost_ret"] = (2.0 * commission_bps + sell_tax_bps + f["slippage_bps_est"]) / 10000.0

    dates = sorted(f["date"].unique())
    prev = {s: 0.0 for s in universe}
    rows: list[dict] = []

    for i, d in enumerate(dates):
        day = f[f["date"] == d].copy()
        target = dict(prev)
        if i % 5 == 0:
            day = day.sort_values("score_smooth", ascending=False).reset_index(drop=True)
            day["rank"] = range(1, len(day) + 1)
            rank_map = {str(r.symbol): int(r.rank) for r in day.itertuples()}
            score_map = {str(r.symbol): float(r.score_smooth) for r in day.itertuples()}
            edge_map = {str(r.symbol): float(2.0 * r.est_cost_ret) for r in day.itertuples()}
            current_hold = {s for s, w in prev.items() if w > 0.0}
            selected = []
            for s in universe:
                rk = rank_map.get(s, 10_000)
                sc = score_map.get(s, 0.0)
                keep = rk <= top_k
                if s in current_hold:
                    keep = keep or (rk <= (top_k + buffer) and sc > 0.0)
                if keep:
                    selected.append(s)
            w = 1.0 / max(1, len(selected))
            target = {s: (w if s in set(selected) else 0.0) for s in universe}

            for s in universe:
                if target.get(s, 0.0) > prev.get(s, 0.0):
                    if score_map.get(s, 0.0) <= edge_map.get(s, 0.0):
                        target[s] = prev.get(s, 0.0)
                if abs(target.get(s, 0.0) - prev.get(s, 0.0)) < no_trade_band:
                    target[s] = prev.get(s, 0.0)

            norm = sum(max(0.0, target[s]) for s in universe)
            if norm > 0:
                for s in universe:
                    target[s] = max(0.0, target[s]) / norm

        for s in universe:
            rows.append({"date": d, "symbol": s, "weight": float(target.get(s, 0.0))})
        prev = target

    return pd.DataFrame(rows)

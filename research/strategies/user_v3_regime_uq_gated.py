from __future__ import annotations

import numpy as np
import pandas as pd

from research.strategies.user_v2_cost_aware import generate_weights as generate_weights_v2


def _avg_pairwise_corr(wide: pd.DataFrame) -> pd.Series:
    vals = []
    idx = []
    for i in range(len(wide)):
        sub = wide.iloc[max(0, i - 19) : i + 1]
        c = sub.corr().replace([np.inf, -np.inf], np.nan)
        if c.shape[0] <= 1:
            vals.append(0.0)
        else:
            m = c.values
            triu = m[np.triu_indices_from(m, k=1)]
            finite = triu[np.isfinite(triu)]
            vals.append(float(finite.mean()) if finite.size else 0.0)
        idx.append(wide.index[i])
    return pd.Series(vals, index=idx)


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    w = generate_weights_v2(frame, universe)
    f = frame.sort_values(["date", "symbol"]).copy()
    f = f.groupby(["date", "symbol"], as_index=False).last()
    f["ret1"] = f.groupby("symbol")["close"].pct_change().fillna(0.0)
    pivot = f.pivot(index="date", columns="symbol", values="ret1").fillna(0.0)
    vol20 = pivot.mean(axis=1).rolling(20).std().fillna(0.0)
    corr20 = _avg_pairwise_corr(pivot)
    uncertainty = pivot.std(axis=1).rolling(20).std().fillna(0.0)
    uq_median = uncertainty.rolling(60, min_periods=20).median().replace(0.0, np.nan)

    out_rows = []
    for d, day in w.groupby("date"):
        weights = {str(r.symbol): float(r.weight) for r in day.itertuples()}
        hist_vol = vol20.loc[:d]
        vol = float(vol20.get(d, 0.0))
        corr = float(corr20.get(d, 0.0))
        panic = vol > float(hist_vol.quantile(0.9)) or corr > 0.75
        risk_off = vol > float(hist_vol.quantile(0.7)) or corr > 0.60

        if panic:
            scale = 0.0
        elif risk_off:
            scale = 0.5
        else:
            scale = 1.0

        cov_window = pivot.loc[:d].tail(20)
        if len(cov_window) >= 2:
            cov = cov_window.cov().fillna(0.0)
            cov_arr = cov.reindex(index=universe, columns=universe, fill_value=0.0).values
        else:
            cov_arr = np.zeros((len(universe), len(universe)), dtype=float)
        w_vec = np.asarray([weights.get(s, 0.0) for s in universe], dtype=float)
        port_vol_ann = float(np.sqrt(max(0.0, w_vec @ cov_arr @ w_vec.T)) * np.sqrt(252.0))
        vol_target_scale = min(1.0, 0.18 / max(1e-8, port_vol_ann))

        uq = float(uncertainty.get(d, 0.0))
        uq_med = (
            float(uq_median.get(d, np.nan))
            if not np.isnan(float(uq_median.get(d, np.nan)))
            else 0.0
        )
        uq_scale = 0.5 if (uq_med > 0 and uq > 1.5 * uq_med) else 1.0

        final_scale = scale * vol_target_scale * uq_scale
        scaled = {s: weights.get(s, 0.0) * final_scale for s in universe}
        norm = sum(max(0.0, v) for v in scaled.values())
        if norm > 0:
            scaled = {s: max(0.0, v) / norm for s, v in scaled.items()}

        for s in universe:
            out_rows.append({"date": d, "symbol": s, "weight": float(scaled.get(s, 0.0))})

    return pd.DataFrame(out_rows)

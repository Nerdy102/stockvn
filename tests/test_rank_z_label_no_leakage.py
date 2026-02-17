from __future__ import annotations

import pandas as pd

from core.ml.targets import compute_rank_z_label, compute_y_excess


def test_rank_z_label_no_leakage() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    for s, base in [("AAA", 10.0), ("BBB", 20.0), ("VNINDEX", 15.0)]:
        for i, d in enumerate(dates):
            rows.append({"symbol": s, "timestamp": d, "close": base + i})
    df = pd.DataFrame(rows)
    out = compute_rank_z_label(compute_y_excess(df, horizon=5), col="y_excess")
    valid = out.dropna(subset=["y_rank_z"]).copy()
    g = valid.groupby(pd.to_datetime(valid["timestamp"]).dt.date)["y_rank_z"]
    assert (g.mean().abs() < 1e-6).all()
    assert ((g.std(ddof=0) - 1.0).abs() < 1e-6).all()

from __future__ import annotations

import pandas as pd

from core.alpha_v3.targets import build_labels_v3


def test_rank_z_label_distribution_per_date() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rows = []
    for sym, base in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0), ("VNINDEX", 15.0)]:
        for i, d in enumerate(dates):
            rows.append({"symbol": sym, "timestamp": d, "close": base + i + (0.1 if sym == "CCC" else 0.0)})

    out = build_labels_v3(pd.DataFrame(rows), horizon=21)
    valid = out.dropna(subset=["y_rank_z"]).copy()
    g = valid.groupby("date")["y_rank_z"]
    assert (g.mean().abs() < 1e-6).all()
    assert ((g.std(ddof=0) - 1.0).abs() < 1e-6).all()

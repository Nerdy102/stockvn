import pandas as pd

from research.strategies import user_v0_current, user_v1_stability_pack


def _toy_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for i, d in enumerate(dates):
        for j, s in enumerate(symbols):
            price = 100 + (i * (j + 1)) % 17 + j
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                }
            )
    return pd.DataFrame(rows)


def _turnover(weights: pd.DataFrame) -> float:
    piv = weights.pivot(index="date", columns="symbol", values="weight").fillna(0.0)
    return float(piv.diff().abs().sum(axis=1).fillna(0.0).sum())


def test_v1_turnover_lower_than_v0_synthetic() -> None:
    frame = _toy_frame()
    universe = sorted(frame["symbol"].unique().tolist())
    w0 = user_v0_current.generate_weights(frame, universe)
    w1 = user_v1_stability_pack.generate_weights(frame, universe)
    assert _turnover(w1) < _turnover(w0)

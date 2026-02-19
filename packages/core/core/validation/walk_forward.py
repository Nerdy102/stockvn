from __future__ import annotations

import statistics


def walk_forward_splits(
    n: int, train_window: int = 252, test_window: int = 63, step: int = 63
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out = []
    i = train_window
    while i + test_window <= n:
        out.append(((i - train_window, i), (i, i + test_window)))
        i += step
    return out


def stability_bucket(std_net_return: float) -> str:
    if std_net_return < 0.05:
        return "Cao"
    if std_net_return <= 0.10:
        return "Vừa"
    return "Thấp"


def summarize_walk_forward(net_returns: list[float]) -> dict[str, float | str]:
    std_val = float(statistics.pstdev(net_returns)) if net_returns else 0.0
    return {"stability_score": std_val, "stability": stability_bucket(std_val)}

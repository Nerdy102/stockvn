from __future__ import annotations


def walk_forward_splits(n: int, train: int, test: int, step: int) -> list[tuple[range, range]]:
    out: list[tuple[range, range]] = []
    start = 0
    while (start + train + test) <= n:
        tr = range(start, start + train)
        te = range(start + train, start + train + test)
        out.append((tr, te))
        start += step
    return out


def purged_cv_splits(
    n: int, k_folds: int, label_horizon_max: int, embargo_days: int
) -> list[tuple[list[int], list[int]]]:
    fold = max(1, n // k_folds)
    splits: list[tuple[list[int], list[int]]] = []
    for k in range(k_folds):
        t0 = k * fold
        t1 = n if k == (k_folds - 1) else min(n, (k + 1) * fold)
        test = list(range(t0, t1))
        train: list[int] = []
        test_min = t0
        test_max = t1 - 1
        for i in range(n):
            overlap = not (i + label_horizon_max < test_min or i > test_max)
            embargo = test_max < i <= (test_max + embargo_days)
            if i in test or overlap or embargo:
                continue
            train.append(i)
        splits.append((train, test))
    return splits

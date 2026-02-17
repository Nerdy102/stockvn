import pandas as pd


def _purged_split(n: int, k: int, horizon: int, embargo: int):
    dates = pd.RangeIndex(n)
    fold = n // k
    for i in range(k):
        lo = i * fold
        hi = n if i == k - 1 else (i + 1) * fold
        test = set(dates[lo:hi])
        train = []
        for t in dates:
            if t in test:
                continue
            if any((t <= u <= t + horizon) for u in test):
                continue
            if hi <= t < hi + embargo:
                continue
            train.append(t)
        yield train, list(test)


def test_purged_kfold_embargo_no_leakage() -> None:
    for train, test in _purged_split(250, 5, horizon=21, embargo=5):
        tset = set(train)
        assert not any(t in tset for t in test)

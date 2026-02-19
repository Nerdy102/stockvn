from __future__ import annotations

import datetime as dt

from core.validation.purged_kfold import PurgedKFold


def test_purged_kfold_no_leakage() -> None:
    n = 120
    timestamps = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    label_end_times = [t + dt.timedelta(days=5) for t in timestamps]

    splitter = PurgedKFold(n_splits=5, embargo_pct=0.02)
    for train_idx, test_idx in splitter.split(timestamps, label_end_times):
        test_start = timestamps[test_idx[0]]
        test_end = timestamps[test_idx[-1]]
        for j in train_idx:
            ts_j = timestamps[j]
            le_j = label_end_times[j]
            assert le_j < test_start or ts_j > test_end

        embargo_len = int((n * 0.02) + 0.999999)
        forbidden = set(range(test_idx[-1] + 1, min(n, test_idx[-1] + 1 + embargo_len)))
        assert not forbidden.intersection(set(train_idx))

from __future__ import annotations

import math


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = max(2, int(n_splits))
        self.embargo_pct = max(0.0, float(embargo_pct))

    def split(self, timestamps: list, label_end_times: list):
        n = len(timestamps)
        if n != len(label_end_times):
            raise ValueError("timestamps và label_end_times phải cùng độ dài")
        fold = n // self.n_splits
        embargo_len = int(math.ceil(n * self.embargo_pct))
        for i in range(self.n_splits):
            lo = i * fold
            hi = n if i == self.n_splits - 1 else (i + 1) * fold
            test_idx = list(range(lo, hi))
            test_start = timestamps[lo]
            test_end = timestamps[hi - 1]

            train_idx = []
            for j in range(n):
                if lo <= j < hi:
                    continue
                ts_j = timestamps[j]
                le_j = label_end_times[j]
                overlap = not (le_j < test_start or ts_j > test_end)
                if overlap:
                    continue
                if hi <= j < min(n, hi + embargo_len):
                    continue
                train_idx.append(j)
            yield train_idx, test_idx

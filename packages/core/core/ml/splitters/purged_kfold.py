from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np


@dataclass
class PurgedKFoldEmbargo:
    n_splits: int = 5
    purge_horizon_days: int = 21
    embargo_days: int = 5

    def split(self, dates: list[dt.date]) -> list[tuple[list[dt.date], list[dt.date]]]:
        unique_dates = sorted(set(dates))
        if not unique_dates:
            return []
        idx = np.arange(len(unique_dates))
        folds = [f.tolist() for f in np.array_split(idx, self.n_splits) if len(f) > 0]

        out: list[tuple[list[dt.date], list[dt.date]]] = []
        for test_idx in folds:
            test_set = set(test_idx)
            test_min = min(test_idx)
            test_max = max(test_idx)
            train_idx: list[int] = []
            for i in idx.tolist():
                if i in test_set:
                    continue
                if (i + self.purge_horizon_days) >= test_min and i <= test_max:
                    continue
                if test_max < i <= (test_max + self.embargo_days):
                    continue
                train_idx.append(i)
            train_dates = [unique_dates[i] for i in train_idx]
            test_dates = [unique_dates[i] for i in test_idx]
            out.append((train_dates, test_dates))
        return out


def validate_no_leakage(
    splits: list[tuple[list[dt.date], list[dt.date]]],
    purge_horizon_days: int,
    embargo_days: int,
) -> bool:
    for train_dates, test_dates in splits:
        if not test_dates:
            continue
        tmin = min(test_dates)
        tmax = max(test_dates)
        for d in train_dates:
            if d in test_dates:
                return False
            dist_to_test_start = (tmin - d).days
            if 0 <= dist_to_test_start <= purge_horizon_days:
                return False
            dist_after_test_end = (d - tmax).days
            if 0 < dist_after_test_end <= embargo_days:
                return False
    return True

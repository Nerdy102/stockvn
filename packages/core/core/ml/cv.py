from __future__ import annotations

import numpy as np
import pandas as pd


def purged_kfold_with_embargo(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int = 5,
    horizon: int = 21,
    embargo: int = 5,
):
    """Date-ordered purged KFold with embargo."""
    idx = np.arange(len(dates))
    folds = np.array_split(idx, n_splits)
    for test_idx in folds:
        tmin, tmax = int(test_idx.min()), int(test_idx.max())
        keep = np.ones(len(idx), dtype=bool)
        keep[test_idx] = False
        for i in idx:
            if not keep[i]:
                continue
            if (i + horizon) >= tmin and i <= tmax:
                keep[i] = False
            if tmax < i <= (tmax + embargo):
                keep[i] = False
        train_idx = idx[keep]
        yield train_idx, test_idx

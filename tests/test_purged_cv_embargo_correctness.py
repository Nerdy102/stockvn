from core.eval_lab.splits import purged_cv_splits


def test_purged_cv_embargo_correctness() -> None:
    splits = purged_cv_splits(n=200, k_folds=8, label_horizon_max=5, embargo_days=5)
    for train, test in splits:
        tset = set(test)
        assert not (set(train) & tset)
        tmin, tmax = min(test), max(test)
        for i in train:
            assert i + 5 < tmin or i > tmax
            assert not (tmax < i <= tmax + 5)

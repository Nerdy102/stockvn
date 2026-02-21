from core.raocmoe.cpd import MeanShiftDetector


def test_grid_bound_and_shift_trigger() -> None:
    det = MeanShiftDetector(
        "mean_shift",
        robust_clip=5.0,
        threshold_base=0.2,
        cooldown_bars=2,
        geometric_base=2,
        max_candidates=64,
    )
    trigger = None
    for t in range(1, 400):
        x = 0.0 if t < 200 else 2.0
        cp, _, _ = det.update(x, t)
        assert len(det.grid.candidates) <= 64
        if cp and trigger is None:
            trigger = t
    assert trigger is not None and trigger < 250

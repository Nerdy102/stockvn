import datetime as dt


def derive_public_date(period_end: dt.date, public_date: dt.date | None) -> dt.date:
    return public_date or (period_end + dt.timedelta(days=45))


def test_feature_point_in_time_public_date() -> None:
    period_end = dt.date(2024, 12, 31)
    assert derive_public_date(period_end, None) == dt.date(2025, 2, 14)

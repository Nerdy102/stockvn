import warnings

import pandas as pd

from research.strategies.user_v3_regime_uq_gated import generate_weights


def test_user_v3_generate_weights_no_runtime_warning_small_sample() -> None:
    rows = []
    for i, d in enumerate(pd.date_range("2025-01-01", periods=3, freq="D")):
        rows.append(
            {
                "date": d,
                "symbol": "AAA",
                "open": 10 + i,
                "high": 10.5 + i,
                "low": 9.5 + i,
                "close": 10 + i,
            }
        )
        rows.append(
            {
                "date": d,
                "symbol": "BBB",
                "open": 20 + i,
                "high": 20.5 + i,
                "low": 19.5 + i,
                "close": 20 + i,
            }
        )
    frame = pd.DataFrame(rows)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        out = generate_weights(frame, ["AAA", "BBB"])
    assert not [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert not out.empty

import datetime as dt

import pandas as pd
from core.corporate_actions import adjust_prices


def test_adjust_prices_hook_returns_frame() -> None:
    bars = pd.DataFrame({"close": [10, 11, 12]})
    out = adjust_prices("AAA", bars, dt.date(2025, 1, 1), dt.date(2025, 1, 31), method="split")
    assert len(out) == 3
    assert "is_adjusted" in out.columns

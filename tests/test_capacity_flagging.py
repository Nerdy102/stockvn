from __future__ import annotations

import pandas as pd

from core.portfolio.analytics import Position
from core.portfolio.dashboard import _capacity


def test_capacity_flagging_toy() -> None:
    positions = {
        "AAA": Position("AAA", quantity=1000, avg_cost=10, market_price=10, market_value=1_000_000_000, unrealized_pnl=0),
    }
    tick_df = pd.DataFrame([{"symbol": "AAA", "adtv_20d": 1_000_000_000.0}])
    out = _capacity(positions, nav=2_000_000_000.0, tick_df=tick_df)
    assert len(out["by_symbol"]) == 1
    assert out["by_symbol"][0]["capacity_value"] == 150_000_000.0
    assert out["by_symbol"][0]["breached"] is True
    assert len(out["flags"]) == 1

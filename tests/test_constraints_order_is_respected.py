from __future__ import annotations

import numpy as np

from core.alpha_v3.portfolio import apply_constraints_strict_order


def test_constraints_order_is_respected() -> None:
    w = np.array([0.40, 0.35, 0.25])
    adtv = np.array([50_000_000.0, 800_000_000.0, 800_000_000.0])
    sectors = ["A", "B", "B"]
    nav = 1_000_000_000.0

    out, cash = apply_constraints_strict_order(w, nav, adtv, sectors, risk_off=False)

    # Step 1 liquidity should bind first name hard (ADTV constraint below 10%)
    liq_cap_0 = adtv[0] * 0.05 * 3.0 / nav
    assert out[0] <= liq_cap_0 + 1e-12

    # Step 2 single-name cap
    assert np.all(out <= 0.10 + 1e-12)

    # Step 3 sector cap
    sec_b = out[1] + out[2]
    assert sec_b <= 0.25 + 1e-12

    # Step 4 min cash 10%
    assert cash >= 0.10 - 1e-12

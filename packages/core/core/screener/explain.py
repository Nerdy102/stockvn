from __future__ import annotations

from typing import Any


def build_screener_explain(
    *,
    filters_passed: dict[str, bool],
    factor_contributions: dict[str, float],
    setup_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "filters": filters_passed,
        "factor_component_contributions": factor_contributions,
        "setups": setup_details,
    }

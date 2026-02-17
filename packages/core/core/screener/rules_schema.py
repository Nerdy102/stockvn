from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScreenerRulesSchema(BaseModel):
    name: str
    universe: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    factor_weights: dict[str, float] = Field(default_factory=dict)
    technical_setups: dict[str, Any] = Field(default_factory=dict)


def validate_rules(payload: dict[str, Any]) -> ScreenerRulesSchema:
    return ScreenerRulesSchema.model_validate(payload)

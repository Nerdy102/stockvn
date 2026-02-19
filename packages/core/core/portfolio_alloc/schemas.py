from __future__ import annotations

from pydantic import BaseModel


class AllocationSuggestion(BaseModel):
    allocator: str
    weights_top: list[dict[str, float | str]]

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from services.api_fastapi.api_fastapi.routers.chart import validate_shapes_json


def test_shapes_schema_validation_rejects_unknown() -> None:
    allowed = json.loads(Path("tests/golden/plotly_shapes_schema_allowed_keys.json").read_text())
    assert "type" in allowed["shape_allowed_keys"]

    bad = [{"type": "line", "x0": "2025-01-01", "x1": "2025-01-02", "y0": 1, "y1": 2, "foo": 123}]
    with pytest.raises(HTTPException):
        validate_shapes_json(bad)

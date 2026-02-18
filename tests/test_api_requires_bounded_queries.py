from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_signals_rejects_unpaginated_large_range() -> None:
    c = TestClient(create_app())
    r = c.get(
        "/signals",
        params={"start": "2020-01-01T00:00:00", "end": "2022-01-05T00:00:00", "offset": 0},
    )
    assert r.status_code == 400
    assert "max range 365" in r.text


def test_fundamentals_rejects_unpaginated_large_range() -> None:
    c = TestClient(create_app())
    r = c.get(
        "/fundamentals",
        params={"start": "2020-01-01", "end": "2022-01-05", "offset": 0},
    )
    assert r.status_code == 400
    assert "max range 365" in r.text

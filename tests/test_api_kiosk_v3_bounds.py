from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_api_kiosk_v3_bounds() -> None:
    c = TestClient(app)
    r1 = c.get('/simple/kiosk_v3', params={'limit_signals': 99})
    assert r1.status_code == 422

    r2 = c.get('/simple/kiosk_v3', params={'lookback': 9999})
    assert r2.status_code == 422

    r3 = c.get('/simple/kiosk_v3', params={'universe': 'BAD'})
    assert r3.status_code == 422

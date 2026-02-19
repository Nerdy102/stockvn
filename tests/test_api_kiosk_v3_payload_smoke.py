from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_api_kiosk_v3_payload_smoke() -> None:
    c = TestClient(app)
    r = c.get('/simple/kiosk_v3', params={'universe': 'VN30', 'limit_signals': 10, 'lookback': 252})
    assert r.status_code == 200
    body = r.json()
    assert 'as_of_date' in body
    assert isinstance(body.get('market_brief_text_vi'), list)
    assert isinstance(body.get('buy_candidates'), list)
    assert isinstance(body.get('sell_candidates'), list)
    assert len(body.get('buy_candidates', [])) <= 10
    assert len(body.get('sell_candidates', [])) <= 10

    readiness = body.get('readiness_summary', {})
    for k in ['stability_score', 'worst_case_net_return', 'worst_case_mdd', 'drift_state', 'kill_switch_state', 'report_id', 'hashes']:
        assert k in readiness

    health = body.get('system_health_summary', {})
    for k in ['db_ok', 'data_freshness_ok', 'last_reconcile_ts']:
        assert k in health

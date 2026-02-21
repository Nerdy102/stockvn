from fastapi.testclient import TestClient

from api_fastapi.main import create_app


def test_run_compare_allows_universe_without_symbols() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.post('/simple/run_compare', json={
            'symbols': [],
            'universe': 'VN30',
            'timeframe': '1D',
            'lookback_days': 252,
            'detail_level': 'tóm tắt'
        })
        assert r.status_code == 200
        assert 'leaderboard' in r.json()

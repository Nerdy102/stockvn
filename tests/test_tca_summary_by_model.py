from fastapi.testclient import TestClient

from api_fastapi.main import create_app


def test_tca_summary_has_by_model_key() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get('/tca/summary', params={'limit': 10})
        assert r.status_code == 200
        body = r.json()
        assert 'by_model' in body

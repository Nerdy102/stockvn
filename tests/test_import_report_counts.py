from __future__ import annotations

from pathlib import Path

from api_fastapi.main import create_app
from core.db.models import Ticker
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel


def test_import_report_counts(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "watchlist_import.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    app = create_app()
    client = TestClient(app)

    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="Tech", industry="Software"))
        s.add(Ticker(symbol="BBB", name="BBB", exchange="HOSE", sector="Bank", industry="Banking"))
        s.commit()

    ws = client.post("/workspaces", json={"user_id": None, "name": "Default"}).json()
    wl = client.post(f"/workspaces/{ws['id']}/watchlists", json={"name": "Core"}).json()

    payload = Path("tests/fixtures/watchlist_import_small.csv").read_bytes()
    r = client.post(
        f"/watchlists/{wl['id']}/import",
        files={"file": ("watchlist_import_small.csv", payload, "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] == 2
    assert body["updated"] == 0
    assert body["invalid_symbols"] == ["BAD"]
    assert body["invalid_rows"] == 1

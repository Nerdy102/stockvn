from __future__ import annotations

from api_fastapi.main import create_app
from core.db.models import Ticker
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel


def test_watchlist_upsert_idempotent(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "watchlist_upsert.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    app = create_app()
    client = TestClient(app)

    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="Tech", industry="Software"))
        s.commit()

    ws = client.post("/workspaces", json={"user_id": None, "name": "Default"}).json()
    wl = client.post(f"/workspaces/{ws['id']}/watchlists", json={"name": "Core"}).json()

    r1 = client.post(
        f"/watchlists/{wl['id']}/items",
        json={"symbol": "AAA", "tags": ["kqkd"], "note": "n1", "pinned": False},
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "inserted"

    r2 = client.post(
        f"/watchlists/{wl['id']}/items",
        json={"symbol": "AAA", "tags": ["policy"], "note": "n2", "pinned": True},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "updated"

    items = client.get(f"/watchlists/{wl['id']}/items").json()
    assert len(items) == 1
    assert items[0]["tags"] == ["policy"]
    assert items[0]["note"] == "n2"
    assert items[0]["pinned"] is True

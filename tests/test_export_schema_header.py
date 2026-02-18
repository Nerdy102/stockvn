from __future__ import annotations

from pathlib import Path

from api_fastapi.main import create_app
from core.db.models import Ticker
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel


def test_export_schema_header(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "watchlist_export.db"
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
    client.post(
        f"/watchlists/{wl['id']}/items",
        json={"symbol": "AAA", "tags": ["kqkd"], "note": "note", "pinned": False},
    )

    r = client.get(f"/watchlists/{wl['id']}/export")
    assert r.status_code == 200
    first_line = r.text.splitlines()[0]
    expected = (
        Path("tests/golden/watchlist_export_header.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_line == expected

import datetime as dt

from sqlmodel import Session

from core.db.models import UniverseSnapshot
from core.db.session import create_db_and_tables, get_engine
from core.universe.manager import UniverseManager


def test_universe_snapshot_membership_used(tmp_path) -> None:
    db = f"sqlite:///{tmp_path}/u.sqlite"
    create_db_and_tables(db)
    engine = get_engine(db)
    with Session(engine) as s:
        s.add(UniverseSnapshot(universe_name="VN30", snapshot_date=dt.date(2025,1,1), symbols_json={"symbols": ["AAA", "BBB"]}))
        s.commit()
        m = UniverseManager(s)
        out = m._snapshot_members_at(dt.date(2025,1,2), "VN30")
        assert out == {"AAA", "BBB"}

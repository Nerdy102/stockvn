from sqlmodel import Session

from core.db.session import create_db_and_tables, get_engine
from core.oms.service import create_draft


def test_create_draft_maps_next_bar_execution(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path}/oms_exec_pref.sqlite"
    create_db_and_tables(db_url)
    with Session(get_engine(db_url)) as s:
        order = create_draft(
            {
                "user_id": "u1",
                "market": "vn",
                "symbol": "FPT",
                "side": "BUY",
                "qty": 10,
                "price": 10000,
                "execution": "thanh nến kế tiếp (next-bar)",
            },
            s,
        )
        assert order.execution_pref == "next_bar"

from __future__ import annotations

import datetime as dt
import importlib

from fastapi.testclient import TestClient
from sqlmodel import Session, select


def test_regression_offline_e2e(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "regression_e2e.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")

    import api_fastapi.main as main_module

    importlib.reload(main_module)
    app = main_module.create_app()

    with TestClient(app) as c:
        h = c.get("/healthz")
        assert h.status_code == 200
        assert h.json().get("status") in {"OK", "FAIL"}

        hd = c.get("/healthz/detail")
        assert hd.status_code == 200
        for k in ["db_ok", "broker_ok", "data_freshness_ok", "kill_switch_state", "last_reconcile_ts"]:
            assert k in hd.json()

        kv3 = c.get("/simple/kiosk_v3", params={"universe": "VN30", "limit_signals": 10, "lookback": 252})
        if kv3.status_code == 404:
            kv3 = c.get("/simple/kiosk", params={"universe": "VN30", "limit_signals": 10, "lookback": 252})
        assert kv3.status_code == 200
        kv3_body = kv3.json()
        assert isinstance(kv3_body.get("market_brief_text_vi"), list)

        cmp = c.post(
            "/simple/run_compare",
            json={
                "symbols": ["FPT", "HPG", "VCB"],
                "timeframe": "1D",
                "lookback_days": 252,
                "detail_level": "chi tiết",
                "engine_version": "v3",
                "market": "vn",
                "trading_type": "spot_paper",
                "include_equity_curve": False,
                "include_trades": False,
                "execution": "giá đóng cửa (close)",
                "enable_bootstrap": True,
                "bootstrap_n_iter": 120,
            },
        )
        assert cmp.status_code == 200
        lb = cmp.json().get("leaderboard", [])
        assert len(lb) == 3
        assert all("config_hash" in x and "dataset_hash" in x and "code_hash" in x for x in lb)

        candidates = (kv3_body.get("buy_candidates") or kv3_body.get("sell_candidates") or [])
        candidate = candidates[0] if candidates else {"symbol": "FPT", "model_id": "model_1", "reason_short": "Tín hiệu demo"}
        side = "BUY" if (kv3_body.get("buy_candidates") or []) else "SELL"

        draft = c.post(
            "/oms/draft",
            json={
                "user_id": "e2e-user",
                "market": "vn",
                "symbol": candidate.get("symbol", "FPT"),
                "timeframe": "1D",
                "mode": "paper",
                "order_type": "limit",
                "side": side,
                "qty": 100,
                "price": 10000,
                "model_id": candidate.get("model_id", "model_1"),
                "config_hash": "regression-e2e",
                "reason_short": candidate.get("reason_short", "Tín hiệu hợp lệ."),
            },
        )
        assert draft.status_code == 200
        order = draft.json()["order"]

        approve = c.post(
            "/oms/approve",
            json={
                "order_id": order["id"],
                "confirm_token": order["confirm_token"],
                "checkboxes": {"risk": True, "edu": True},
            },
        )
        assert approve.status_code == 200

        execute = c.post(
            "/oms/execute",
            json={
                "order_id": order["id"],
                "data_freshness": {"as_of_date": dt.date.today().isoformat()},
                "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 1_999_000_100.0, "orders_today": 0},
                "drift_alerts": {"drift_paused": False, "kill_switch_on": False},
            },
        )
        assert execute.status_code == 200
        assert execute.json()["order"]["status"] == "FILLED"

        timeline = c.get(f"/oms/orders/{order['id']}")
        assert timeline.status_code == 200

        from core.db.session import get_engine
        from core.oms.models import Fill, OrderEvent, PortfolioSnapshot
        from core.settings import get_settings

        settings = get_settings()
        with Session(get_engine(settings.DATABASE_URL)) as db:
            events = db.exec(select(OrderEvent).where(OrderEvent.order_id == order["id"]).order_by(OrderEvent.ts.asc())).all()
            statuses = [e.to_status for e in events]
            assert "DRAFT" in statuses
            assert "APPROVED" in statuses
            assert "SENT" in statuses or "ACKED" in statuses
            assert "FILLED" in statuses

            fill_rows = db.exec(select(Fill).where(Fill.order_id == order["id"]).order_by(Fill.ts.asc())).all()
            assert len(fill_rows) >= 1

            latest_snapshot = db.exec(select(PortfolioSnapshot).order_by(PortfolioSnapshot.ts.desc())).first()
            assert latest_snapshot is not None
            assert latest_snapshot.nav_est >= 0

        c.post("/controls/kill_switch/on", json={})
        blocked = c.post(
            "/oms/execute",
            json={
                "order_id": order["id"],
                "data_freshness": {"as_of_date": dt.date.today().isoformat()},
                "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 1_999_000_100.0, "orders_today": 0},
            },
        )
        assert blocked.status_code == 403
        assert blocked.json()["detail"]["reason_code"] == "KILL_SWITCH_ON"

        from jobs.reconcile import run_reconciliation

        settings = get_settings()
        with Session(get_engine(settings.DATABASE_URL)) as db:
            report = run_reconciliation(db)
        assert report.status == "OK"

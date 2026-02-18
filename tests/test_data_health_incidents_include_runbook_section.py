from __future__ import annotations

import datetime as dt

from core.db.models import DataHealthIncident, PriceOHLCV
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler.jobs import compute_data_quality_metrics_job


def test_incidents_include_runbook_section() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(
            DataHealthIncident(
                source="schema_monitor",
                severity="HIGH",
                status="OPEN",
                summary="prev",
                details_json={"schema_keys": ["symbol", "timeframe", "timestamp", "open"]},
                runbook_section="RB-SCHEMA-DIFF",
                suggested_actions_json={"actions": ["a"]},
                created_at=dt.datetime(2025, 1, 1),
            )
        )
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime(2025, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10,
                volume=1000,
                value_vnd=10000,
                source="test",
                quality_flags={},
            )
        )
        s.commit()

        compute_data_quality_metrics_job(s)

        inc = s.exec(select(DataHealthIncident).where(DataHealthIncident.source == "schema_monitor").order_by(DataHealthIncident.id.desc())).first()
        assert inc is not None
        assert inc.runbook_section == "RB-SCHEMA-DIFF"
        assert isinstance(inc.suggested_actions_json, dict)
        assert "actions" in inc.suggested_actions_json

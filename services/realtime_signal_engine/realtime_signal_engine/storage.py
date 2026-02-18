from __future__ import annotations

import datetime as dt

from core.db.models import (
    AlertAction,
    AlertCooldown,
    AlertIntraday,
    NotificationLog,
    SignalIntraday,
)
from sqlmodel import Session, select


class SignalStorage:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_signal(self, row: dict[str, object]) -> bool:
        symbol = str(row["symbol"])
        tf = str(row["timeframe"])
        end_ts = dt.datetime.fromisoformat(str(row["end_ts"]).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
        exists = self.session.exec(
            select(SignalIntraday)
            .where(SignalIntraday.symbol == symbol)
            .where(SignalIntraday.timeframe == tf)
            .where(SignalIntraday.end_ts == end_ts)
        ).first()
        if exists is not None:
            return False
        self.session.add(
            SignalIntraday(
                symbol=symbol,
                timeframe=tf,
                end_ts=end_ts,
                indicators_json=dict(row.get("indicators_json", {})),
                setups_json=dict(row.get("setups_json", {})),
                signal_hash=str(row.get("signal_hash", "")),
            )
        )
        self.session.commit()
        return True

    def cooldown_allows(
        self, symbol: str, tf: str, rule_key: str, bar_index: int, cooldown_bars: int
    ) -> bool:
        row = self.session.exec(
            select(AlertCooldown)
            .where(AlertCooldown.symbol == symbol)
            .where(AlertCooldown.timeframe == tf)
            .where(AlertCooldown.rule_key == rule_key)
        ).first()
        if row is None:
            return True
        return int(bar_index - row.last_bar_index) >= int(cooldown_bars)

    def mark_cooldown(self, symbol: str, tf: str, rule_key: str, bar_index: int) -> None:
        row = self.session.exec(
            select(AlertCooldown)
            .where(AlertCooldown.symbol == symbol)
            .where(AlertCooldown.timeframe == tf)
            .where(AlertCooldown.rule_key == rule_key)
        ).first()
        if row is None:
            row = AlertCooldown(
                symbol=symbol, timeframe=tf, rule_key=rule_key, last_bar_index=bar_index
            )
        else:
            row.last_bar_index = bar_index
            row.updated_at = dt.datetime.utcnow()
        self.session.add(row)
        self.session.commit()

    def emit_alert(self, payload: dict[str, object]) -> None:
        symbol = str(payload["symbol"])
        tf = str(payload["timeframe"])
        end_ts = dt.datetime.fromisoformat(str(payload["end_ts"]).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
        alert = AlertIntraday(
            symbol=symbol,
            timeframe=tf,
            end_ts=end_ts,
            alert_key=str(payload["alert_key"]),
            severity=int(payload.get("severity", 1)),
            payload_json=dict(payload),
        )
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)
        self.session.add(
            AlertAction(alert_id=int(alert.id or 0), action="TRIGGERED", payload_json=dict(payload))
        )
        self.session.add(
            NotificationLog(kind="alerts_intraday", channel="internal", payload_json=dict(payload))
        )
        self.session.commit()

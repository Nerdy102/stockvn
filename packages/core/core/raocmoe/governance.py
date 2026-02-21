from __future__ import annotations

import datetime as dt
import json
from typing import Any

from .types import GovernanceStatus


class GovernanceEngine:
    def __init__(self, cfg: dict, redis_client: Any | None = None) -> None:
        self.cfg = cfg["gov"]
        self.redis = redis_client
        self.status = GovernanceStatus(
            paused=bool(self.cfg["paused_default"]),
            pause_reason=None,
            last_change_ts=dt.datetime.utcnow().isoformat() + "Z",
            details={},
        )
        self.audit_log: list[dict[str, Any]] = []

    def _set_status(
        self, paused: bool, reason: str | None, details: dict[str, Any]
    ) -> GovernanceStatus:
        now = dt.datetime.utcnow().isoformat() + "Z"
        self.status = GovernanceStatus(
            paused=paused,
            pause_reason=reason,
            last_change_ts=now,
            details={k: details[k] for k in sorted(details)},
        )
        self.audit_log.append(
            {
                "ts": now,
                "paused": paused,
                "reason_code": reason,
                "details": self.status.details,
            }
        )
        self._persist()
        return self.status

    def evaluate(
        self,
        *,
        uq_undercoverage: bool,
        psi: float,
        tca_shift: bool,
        data_stale: bool,
        extra: dict[str, Any] | None = None,
    ) -> GovernanceStatus:
        details = dict(extra or {})
        details["psi"] = float(psi)
        if uq_undercoverage:
            details["trigger"] = "UQ_UNDERCOVERAGE"
            return self._set_status(True, "UQ_UNDERCOVERAGE", details)
        if psi > 0.25:
            details["trigger"] = "DRIFT_HIGH"
            return self._set_status(True, "DRIFT_HIGH", details)
        if tca_shift:
            details["trigger"] = "TCA_REGIME_SHIFT"
            return self._set_status(True, "TCA_REGIME_SHIFT", details)
        if data_stale:
            details["trigger"] = "DATA_STALE"
            return self._set_status(True, "DATA_STALE", details)
        details["trigger"] = "CLEAR"
        return self._set_status(False, None, details)

    def derisk_scale(self, regime: str) -> float:
        if regime == "RISK_OFF":
            return float(self.cfg["derisk_scale"]["risk_off"])
        if regime == "PANIC_VOL":
            return float(self.cfg["derisk_scale"]["panic_vol"])
        return 1.0

    def _persist(self) -> None:
        if self.redis is None:
            return
        payload = json.dumps(self.status.to_json(), sort_keys=True, separators=(",", ":"))
        if hasattr(self.redis, "set"):
            self.redis.set("realtime:raocmoe:gov_status", payload)

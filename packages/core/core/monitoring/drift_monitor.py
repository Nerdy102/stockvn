from __future__ import annotations

import datetime as dt
import json
from typing import Any

from sqlmodel import Field, SQLModel


class DriftAlert(SQLModel, table=True):
    __tablename__ = "drift_alert_trade"

    id: int | None = Field(default=None, primary_key=True)
    ts: dt.datetime = Field(default_factory=dt.datetime.utcnow, index=True)
    model_id: str = Field(index=True)
    market: str = Field(index=True)
    severity: str = Field(index=True)
    code: str = Field(index=True)
    message: str
    snapshot_json: str = "{}"


# Tương thích ngược
DriftAlertTrade = DriftAlert


def evaluate_drift_pause(
    *,
    model_id: str,
    market: str,
    recent_trades: list[dict[str, Any]],
) -> list[DriftAlert]:
    if len(recent_trades) < 10:
        return []
    out: list[DriftAlert] = []

    t10 = recent_trades[-10:]
    t30 = recent_trades[-30:]
    slip_real = sum(float(x.get("realized_slippage", 0.0)) for x in t10)
    slip_est = sum(float(x.get("est_slippage", 0.0)) for x in t10)
    slip_ratio = slip_real / max(slip_est, 1e-9)

    wins = sum(1 for x in t30 if float(x.get("pnl", 0.0)) > 0)
    hit_rate = wins / max(len(t30), 1)
    avg_pnl = sum(float(x.get("pnl", 0.0)) for x in t30) / max(len(t30), 1)

    mdd7 = [float(x.get("mdd", 0.0)) for x in recent_trades[-7:]]
    mdd_jump = (mdd7[-1] - mdd7[0]) if len(mdd7) >= 2 else 0.0

    if slip_ratio > 2.0:
        out.append(
            DriftAlert(
                model_id=model_id,
                market=market,
                severity="HIGH",
                code="DRIFT_SLIPPAGE_RATIO_HIGH",
                message="Tỷ lệ trượt giá thực tế/ước tính vượt ngưỡng 2.0 trong 10 lệnh gần nhất.",
                snapshot_json=json.dumps({"slippage_ratio": slip_ratio}, ensure_ascii=False),
            )
        )
    if model_id in {"model_1", "model_2"} and len(t30) >= 30 and hit_rate < 0.35:
        out.append(
            DriftAlert(
                model_id=model_id,
                market=market,
                severity="HIGH",
                code="DRIFT_HIT_RATE_LOW",
                message="Tỷ lệ thắng dưới 35% trong 30 lệnh gần nhất.",
                snapshot_json=json.dumps({"hit_rate": hit_rate}, ensure_ascii=False),
            )
        )
    if len(t30) >= 30 and avg_pnl < 0 and mdd_jump > 0.03:
        out.append(
            DriftAlert(
                model_id=model_id,
                market=market,
                severity="HIGH",
                code="DRIFT_MDD_ACCELERATING",
                message="P&L trung bình âm và MDD rolling tăng nhanh >3% trong 7 ngày.",
                snapshot_json=json.dumps(
                    {"avg_trade_pnl": avg_pnl, "mdd_jump_7d": mdd_jump}, ensure_ascii=False
                ),
            )
        )

    return out

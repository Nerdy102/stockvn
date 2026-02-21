from __future__ import annotations

import json
import os
from typing import Any

import httpx
from redis import Redis


def _latest_close(
    redis_client: Redis, symbol: str = "BTCUSDT", tf: str = "15m"
) -> tuple[float, str]:
    key = f"realtime:bars:{symbol}:{tf}"
    rows = redis_client.lrange(key, -1, -1)
    if not rows:
        raise RuntimeError(f"No realtime bars found in {key}. Start ingestor+bar_builder first.")
    row = json.loads(rows[-1])
    return float(row["c"]), str(row.get("end_ts", ""))


def _last_update_ts(redis_client: Redis) -> str:
    raw = redis_client.get("realtime:ops:summary")
    if not raw:
        return ""
    payload = json.loads(raw)
    return str(payload.get("last_update", ""))


def main() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    redis_client = Redis.from_url(redis_url, decode_responses=True)
    px, bar_end_ts = _latest_close(redis_client)
    as_of_ts = _last_update_ts(redis_client) or bar_end_ts

    portfolio_snapshot: dict[str, Any] = {
        "cash": 1_000_000.0,
        "nav_est": 1_000_000.0,
        "orders_today": 0,
    }

    with httpx.Client(timeout=30.0) as client:
        draft = client.post(
            f"{api_base_url}/oms/draft",
            json={
                "user_id": "crypto-demo",
                "market": "crypto",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "mode": "paper",
                "order_type": "limit",
                "side": "BUY",
                "qty": 0.001,
                "price": px,
                "model_id": "realtime-demo",
                "config_hash": "crypto-rt-demo",
                "reason_short": "Realtime BTC paper demo",
            },
        )
        draft.raise_for_status()
        order = draft.json()["order"]

        approve = client.post(
            f"{api_base_url}/oms/approve",
            json={
                "order_id": order["id"],
                "confirm_token": order["confirm_token"],
                "checkboxes": {"risk": True, "edu": True},
            },
        )
        approve.raise_for_status()

        execute = client.post(
            f"{api_base_url}/oms/execute",
            json={
                "order_id": order["id"],
                "data_freshness": {"as_of_ts": as_of_ts},
                "portfolio_snapshot": portfolio_snapshot,
                "drift_alerts": {"drift_paused": False, "kill_switch_on": False},
            },
        )
        execute.raise_for_status()
        out = execute.json()

    filled = out.get("order", {})
    print("OMS realtime paper execution completed")
    print(f"order_id={filled.get('id')}")
    print(f"status={filled.get('status')}")
    print(f"symbol={filled.get('symbol')} price={filled.get('price')} qty={filled.get('qty')}")


if __name__ == "__main__":
    main()

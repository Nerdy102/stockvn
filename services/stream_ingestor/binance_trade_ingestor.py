from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
from typing import Any

from redis import Redis
from websockets.asyncio.client import connect

STREAM_KEY = "stream:market_events"


def _to_iso8601_z(ts_ms: int) -> str:
    return (
        dt.datetime.fromtimestamp(ts_ms / 1000.0, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _to_market_event(trade_payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(trade_payload["s"]).upper()
    trade_id = str(trade_payload["t"])
    provider_ts = _to_iso8601_z(int(trade_payload["T"]))
    base = {
        "event_type": "TRADE",
        "event_id": f"binance:{symbol}:{trade_id}",
        "symbol": symbol,
        "provider_ts": provider_ts,
        "price": float(trade_payload["p"]),
        "qty": float(trade_payload["q"]),
    }
    payload_hash = hashlib.sha256(_canonical_json(base).encode("utf-8")).hexdigest()
    return {**base, "payload_hash": payload_hash}


def _ws_url(symbols: list[str]) -> str:
    streams = [f"{s.lower()}@trade" for s in symbols]
    if len(streams) == 1:
        return f"wss://stream.binance.com:9443/ws/{streams[0]}"
    return f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"


def _extract_trade_payload(msg: dict[str, Any]) -> dict[str, Any] | None:
    if "stream" in msg and isinstance(msg.get("data"), dict):
        data = msg["data"]
    else:
        data = msg
    if str(data.get("e", "")).lower() != "trade":
        return None
    return data


async def run_forever(symbols: list[str], redis_url: str) -> None:
    log = logging.getLogger(__name__)
    redis = Redis.from_url(redis_url, decode_responses=True)
    url = _ws_url(symbols)

    while True:
        try:
            async with connect(url, ping_interval=20, ping_timeout=20) as ws:
                log.info("Connected Binance WS: %s", url)
                async for raw in ws:
                    msg = json.loads(raw)
                    trade = _extract_trade_payload(msg)
                    if trade is None:
                        continue
                    event = _to_market_event(trade)
                    redis.xadd(
                        STREAM_KEY,
                        {"payload": _canonical_json(event)},
                        maxlen=200_000,
                        approximate=True,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Binance ingestor error: %s. reconnecting in 2s", exc)
            await asyncio.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance public trade ingestor -> Redis stream")
    parser.add_argument(
        "--symbols", default="BTCUSDT", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT"
    )
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    asyncio.run(run_forever(symbols=symbols, redis_url=args.redis))


if __name__ == "__main__":
    main()

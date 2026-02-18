from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from typing import Any

import pandas as pd

from indicators.incremental import IndicatorState, update_indicators_state

from .evaluator import evaluate_alert_dsl_on_bar_close, evaluate_setups, governance_paused_flag
from .state_store import StateStore
from .storage import SignalStorage


class RealtimeSignalEngine:
    def __init__(
        self,
        *,
        redis_client: Any,
        storage: SignalStorage,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.redis = redis_client
        self.storage = storage
        self.state = StateStore(redis_client)
        self.config = config or {}

    def _read_bar_close_stream(self, tf: str) -> list[dict[str, Any]]:
        rows = self.redis.xrange(f"stream:bar_close:{tf}")
        out: list[dict[str, Any]] = []
        for _, fields in rows:
            payload = fields.get("payload")
            if isinstance(payload, str):
                out.append(json.loads(payload))
            elif isinstance(payload, dict):
                out.append(payload)
        return out

    def _bootstrap_history(
        self, symbol: str, tf: str, existing: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if existing:
            return existing
        return self.state.get_hot_cache_bars(symbol, tf)

    def run_once(self) -> dict[str, int]:
        metrics = {"signals": 0, "alerts": 0}
        now = dt.datetime.utcnow()

        for tf in ["15m", "60m"]:
            bars = self._read_bar_close_stream(tf)
            histories: dict[str, list[dict[str, Any]]] = {}
            for i, bar in enumerate(bars):
                symbol = str(bar["symbol"])
                hist = histories.get(symbol, [])
                hist = self._bootstrap_history(symbol, tf, hist)
                hist.append(bar)
                histories[symbol] = hist[-200:]

                st = self.state.get_indicator_state(symbol, tf)
                if st is None:
                    st = IndicatorState()
                    # bootstrap incrementally from cached bars (if state missing)
                    for hb in histories[symbol][:-1]:
                        end_ts = dt.datetime.fromisoformat(str(hb["end_ts"]).replace("Z", "+00:00"))
                        _, st = update_indicators_state(
                            st,
                            end_ts=end_ts,
                            open_=float(hb["o"]),
                            high=float(hb["h"]),
                            low=float(hb["l"]),
                            close=float(hb["c"]),
                            volume=float(hb["v"]),
                        )

                end_ts = dt.datetime.fromisoformat(str(bar["end_ts"]).replace("Z", "+00:00"))
                indicators, st = update_indicators_state(
                    st,
                    end_ts=end_ts,
                    open_=float(bar["o"]),
                    high=float(bar["h"]),
                    low=float(bar["l"]),
                    close=float(bar["c"]),
                    volume=float(bar["v"]),
                )
                self.state.set_indicator_state(symbol, tf, st)

                hdf = pd.DataFrame(histories[symbol])
                if "close" not in hdf.columns and "c" in hdf.columns:
                    hdf = hdf.rename(
                        columns={"c": "close", "h": "high", "l": "low", "v": "volume", "o": "open"}
                    )
                setups = evaluate_setups(hdf, indicators)
                paused = governance_paused_flag(self.config)
                snapshot = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "end_ts": str(bar["end_ts"]),
                    "indicators_json": indicators,
                    "setups_json": setups,
                    "order_generation_enabled": not paused,
                }
                snapshot["signal_hash"] = sha256(
                    json.dumps(
                        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()

                if self.storage.upsert_signal(snapshot):
                    metrics["signals"] += 1
                self.state.set_signal_snapshot(symbol, tf, snapshot)

                expression = str(self.config.get("alert_expression", "close > EMA20"))
                eval_df = hdf.copy()
                for k, v in indicators.items():
                    eval_df[k] = float(v)
                should_alert = evaluate_alert_dsl_on_bar_close(eval_df, expression)
                if should_alert and self.storage.cooldown_allows(symbol, tf, expression, i, 26):
                    payload = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "end_ts": str(bar["end_ts"]),
                        "alert_key": expression,
                        "severity": 1,
                        "setups": setups,
                    }
                    self.storage.emit_alert(payload)
                    self.storage.mark_cooldown(symbol, tf, expression, i)
                    metrics["alerts"] += 1
                    if setups.get("volume_spike", False):
                        self.state.push_hot("volume_spike", payload)

        self.state.set_ops_summary({"last_update": now.isoformat() + "Z", "lag": 0, **metrics})
        return metrics

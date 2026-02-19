from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any


def _redact_text(v: str) -> str:
    text = str(v)
    import os

    for key in ["SSI_CONSUMER_ID", "SSI_CONSUMER_SECRET", "SSI_PRIVATE_KEY_PATH", "CRYPTO_API_KEY", "CRYPTO_SECRET"]:
        raw = os.getenv(key, "")
        if raw and raw in text:
            text = text.replace(raw, "***REDACTED***")
    return text


_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


@contextmanager
def request_id_context(request_id: str):
    token = _request_id_ctx.set(request_id)
    try:
        yield
    finally:
        _request_id_ctx.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        for key in [
            "module",
            "symbol",
            "timeframe",
            "provider",
            "latency_ms",
            "job_id",
            "request_id",
            "correlation_id",
            "reason_code",
        ]:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        request_id = _request_id_ctx.get()
        if request_id and "request_id" not in payload:
            payload["request_id"] = request_id
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(lvl)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(lvl)


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    return logging.LoggerAdapter(logging.getLogger(name), context)
